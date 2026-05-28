from __future__ import annotations

import time
from typing import List

import cv2

from camera_guidance import compute_center_alignment
from detection import Detection, make_detector
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


def main() -> None:
    args = build_arg_parser().parse_args()
    source = make_source(args)
    detector = make_detector(args.detector)
    overlap_guide = FaceOverlapGuidance(
        overlap_threshold=args.overlap_threshold,
        speech=SpeechNotifier(enabled=not args.disable_overlap_voice),
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
        "Overlap guidance:",
        "visual only" if args.disable_overlap_voice else "visual + voice",
        f"(threshold={args.overlap_threshold})",
    )
    print("Modes: 1=manual, 2=person1, 3=count(N), 4=ratio")
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

            overlap_tracks, overlap_events = overlap_guide.process(
                analysis_frame,
                detections,
                speak=not args.disable_overlap_voice,
            )
            person_count = len(detections)
            area_ratio = compute_people_area_ratio(detections, analysis_frame.shape)
            center_alignment = compute_center_alignment(
                detections,
                analysis_frame.shape,
                args.center_tolerance,
            )

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
            auto_triggered = (
                mode != MODE_MANUAL
                and stable_hits >= args.stable_frames
                and (now - last_capture_at) >= args.cooldown_sec
            )
            if auto_triggered:
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
