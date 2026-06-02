from __future__ import annotations

import time
from typing import List

import cv2

from camera_guidance import CenterAlignmentVoiceGuidance, compute_center_alignment
from countdown import STATUS_CAPTURE, STATUS_COUNTING, CountdownController
from detection import Detection, make_detector
from outofframe_guidance import FaceOutOfFrameGuidance
from overlap_guidance import FaceOverlapGuidance
from speech import SpeechNotifier
from shot_capture_experiment import (
    MODE_MANUAL,
    MODE_PERSON1,
    MODE_COUNT3,
    MODE_RATIO,
    MODES,
    OUTPUT_ROOT,
    WINDOW_NAME,
    CaptureRecord,
    append_csv,
    build_arg_parser,
    clamp_crop,
    compute_people_area_ratio,
    draw_overlay,
    ensure_session_dirs,
    make_source,
    mode_condition_met,
    save_capture,
)


def draw_countdown_overlay(overlay, remaining: int) -> None:
    """촬영 카운트다운 숫자를 화면 중앙에 크게 표시합니다."""
    if remaining <= 0:
        return
    height, width = overlay.shape[:2]
    text = str(remaining)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 6.0
    thickness = 12
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    org = ((width - text_w) // 2, (height + text_h) // 2)
    # 검은 외곽선 + 흰 글자로 어떤 배경에서도 보이게 한다.
    cv2.putText(overlay, text, org, font, scale, (0, 0, 0), thickness + 8, cv2.LINE_AA)
    cv2.putText(overlay, text, org, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> None:
    args = build_arg_parser().parse_args()
    source = make_source(args)
    detector = make_detector(args.detector)
    speech = SpeechNotifier(enabled=not args.disable_guidance_voice)
    overlap_guide = FaceOverlapGuidance(
        overlap_threshold=args.overlap_threshold,
        speech=speech,
    )
    outofframe_guide = FaceOutOfFrameGuidance(
        speech=speech,
    )
    center_voice_guide = CenterAlignmentVoiceGuidance(
        speech=speech,
    )
    countdown = CountdownController(
        speech=speech,
        steps=args.countdown_steps,
        step_sec=args.countdown_step_sec,
        enabled=not args.no_countdown,
    )
    target_persons = max(1, args.target_persons)

    session_name = time.strftime("%Y%m%d-%H%M%S")
    session_dir = OUTPUT_ROOT / session_name
    session_dirs = ensure_session_dirs(session_dir)
    csv_path = session_dir / "captures.csv"

    mode = args.start_mode
    stable_hits = 0
    last_capture_at = 0.0
    captures_per_mode = {name: 0 for name in MODES}
    frame_index = 0
    session_started_at = time.time()
    last_detections: List[Detection] = []

    print("=" * 68)
    print("Session:", session_dir)
    print("Environment: conda activate cv")
    print(f"Detector: {detector.name}   Target persons: {target_persons}")
    print(
        "Guidance:",
        "visual only" if args.disable_guidance_voice else "visual + voice",
        f"(threshold={args.overlap_threshold})",
    )
    print("Modes: 1=manual, 2=person1, 3=count(N), 4=ratio")
    count3_quality = "off (head-count only)" if args.disable_count3_quality else "on (inside+center+no-overlap)"
    print(f"count3 quality gate: {count3_quality}")
    countdown_state = "off" if args.no_countdown else f"{args.countdown_steps} steps x {args.countdown_step_sec}s"
    print(f"Capture countdown: {countdown_state}")
    print("Target persons: '-' decrease, '=' increase")
    print("Capture: c or SPACE   Quit: q")
    print("=" * 68)

    try:
        while True:
            frame = source.read()
            if frame is None:
                break

            analysis_frame, crop_rect = clamp_crop(frame, args.crop)

            if frame_index % max(1, args.detect_every) == 0:
                last_detections = detector.detect(analysis_frame)
            detections = last_detections

            # 카운트다운 중에는 안내 음성을 억제해 "하나·둘·셋"이 묻히지 않게 한다.
            voice_allowed = not args.disable_guidance_voice and not countdown.active

            _, outofframe_events = outofframe_guide.process(
                analysis_frame,
                detections,
                speak=voice_allowed,
            )
            overlap_tracks, overlap_events = overlap_guide.process(
                analysis_frame,
                detections,
                speak=voice_allowed and not outofframe_events,
            )
            person_count = len(detections)
            area_ratio = compute_people_area_ratio(detections, analysis_frame.shape)
            center_alignment = compute_center_alignment(
                detections,
                analysis_frame.shape,
                args.center_tolerance,
            )
            center_voice_guide.process(
                center_alignment,
                speak=(
                    voice_allowed
                    and not outofframe_events
                    and not overlap_events
                ),
            )

            # 세 품질조건: 화면내부(잘림 없음) / 겹침 없음 / 중심 정렬.
            inside_frame = not outofframe_events
            no_overlap = not overlap_events
            centered = center_alignment.has_target and center_alignment.aligned

            condition_met = mode_condition_met(
                mode=mode,
                detections=detections,
                frame_shape=analysis_frame.shape,
                person_count=person_count,
                area_ratio=area_ratio,
                target_persons=target_persons,
                ratio_min=args.ratio_min,
                ratio_max=args.ratio_max,
                one_person_ratio_min=args.one_person_ratio_min,
                one_person_ratio_max=args.one_person_ratio_max,
                center_tolerance=args.center_tolerance,
                border_margin_ratio=args.border_margin_ratio,
                inside_frame=inside_frame,
                no_overlap=no_overlap,
                centered=centered,
                require_quality=not args.disable_count3_quality,
            )

            if mode == MODE_MANUAL:
                stable_hits = 0
            elif condition_met:
                stable_hits += 1
            else:
                stable_hits = 0

            overlay = draw_overlay(
                frame=frame,
                crop_rect=crop_rect,
                detections=detections,
                overlap_tracks=overlap_tracks,
                overlap_events=overlap_events,
                mode=mode,
                condition_met=condition_met,
                person_count=person_count,
                area_ratio=area_ratio,
                stable_hits=stable_hits,
                stable_frames=args.stable_frames,
                captures_per_mode=captures_per_mode,
                target_persons=target_persons,
                detector_name=detector.name,
                center_alignment=center_alignment,
            )

            now = time.time()
            # 촬영 준비 = 자동 모드 + 모드 조건 충족 + 안정 프레임 + 쿨다운 경과.
            ready = (
                mode != MODE_MANUAL
                and condition_met
                and stable_hits >= args.stable_frames
                and (now - last_capture_at) >= args.cooldown_sec
            )
            # 준비되면 "하나·둘·셋" 카운트다운을 진행하고, 끝나면 촬영 신호를 받는다.
            # 카운트다운 도중 조건이 깨지면 STATUS_ABORTED 로 자동 취소된다.
            capture_status = countdown.update(ready, now)
            if capture_status == STATUS_CAPTURE:
                captures_per_mode[mode] += 1
                capture_id = f"{mode}_{captures_per_mode[mode]:03d}"
                record = CaptureRecord(
                    capture_id=capture_id,
                    mode=mode,
                    trigger="auto",
                    timestamp=now,
                    elapsed_sec=now - session_started_at,
                    person_count=person_count,
                    people_area_ratio=area_ratio,
                    stable_hits=stable_hits,
                    raw_path=f"{capture_id}_raw.png",
                    overlay_path=f"{capture_id}_overlay.png",
                )
                save_capture(session_dirs, record, frame, overlay)
                append_csv(csv_path, record)
                last_capture_at = now
                stable_hits = 0

            if capture_status == STATUS_COUNTING:
                draw_countdown_overlay(overlay, countdown.remaining_steps(now))

            cv2.imshow(WINDOW_NAME, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("1"):
                mode = MODE_MANUAL
                stable_hits = 0
            elif key == ord("2"):
                mode = MODE_PERSON1
                stable_hits = 0
            elif key == ord("3"):
                mode = MODE_COUNT3
                stable_hits = 0
            elif key == ord("4"):
                mode = MODE_RATIO
                stable_hits = 0
            elif key == ord("-"):
                target_persons = max(1, target_persons - 1)
                stable_hits = 0
                print(f"[target] persons = {target_persons}")
            elif key in (ord("="), ord("+")):
                target_persons += 1
                stable_hits = 0
                print(f"[target] persons = {target_persons}")
            elif key in (ord("c"), 32):
                captures_per_mode[mode] += 1
                now = time.time()
                capture_id = f"{mode}_{captures_per_mode[mode]:03d}"
                record = CaptureRecord(
                    capture_id=capture_id,
                    mode=mode,
                    trigger="manual-key",
                    timestamp=now,
                    elapsed_sec=now - session_started_at,
                    person_count=person_count,
                    people_area_ratio=area_ratio,
                    stable_hits=stable_hits,
                    raw_path=f"{capture_id}_raw.png",
                    overlay_path=f"{capture_id}_overlay.png",
                )
                save_capture(session_dirs, record, frame, overlay)
                append_csv(csv_path, record)
                last_capture_at = now

            frame_index += 1
    finally:
        source.close()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
