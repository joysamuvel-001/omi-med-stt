SAME_SPEAKER_MERGE_GAP = 0.5

def resolve_overlaps(segments: list) -> list:
    """
    Ensure no segments overlap in time.
    If a segment is completely inside another, split the larger segment.
    If they partially overlap, split the difference at the midpoint.
    """
    if not segments:
        return []
    
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    resolved = []
    
    for seg in sorted_segs:
        if not resolved:
            resolved.append(dict(seg))
            continue
        
        prev = resolved[-1]
        
        # If no overlap, just append
        if seg["start"] >= prev["end"]:
            resolved.append(dict(seg))
            continue
            
        # Overlap detected!
        # Case 1: seg is completely inside prev
        if seg["end"] <= prev["end"]:
            part1_end = seg["start"]
            part2_start = seg["end"]
            
            prev_copy = dict(prev)
            
            # Shrink prev to end when seg starts
            prev["end"] = part1_end
            
            # Append seg
            resolved.append(dict(seg))
            
            # Append the remainder of prev if it has valid duration
            if prev_copy["end"] - part2_start >= 0.2:
                part2 = dict(prev_copy)
                part2["start"] = part2_start
                resolved.append(part2)
                
        # Case 2: Partial overlap
        else:
            midpoint = (seg["start"] + prev["end"]) / 2
            prev["end"] = midpoint
            seg["start"] = midpoint
            resolved.append(dict(seg))
            
    # Filter out segments that became too short
    final_segs = []
    for s in resolved:
        if s["end"] - s["start"] >= 0.2:
            final_segs.append(s)
            
    final_segs.sort(key=lambda s: s["start"])
    return final_segs


def process_diarization(raw_segments: list) -> list:
    if not raw_segments:
        return []

    parsed = []
    for seg in raw_segments:
        # --- FIX: handle Sortformer strings, tuples, AND dicts ---
        if isinstance(seg, str):
            parts = seg.strip().split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end   = float(parts[1])
            except ValueError:
                print(f"[diarization] skipping malformed segment: {seg!r}")
                continue
            label = parts[2]
        elif isinstance(seg, (tuple, list)):
            print(f"[diarization] DEBUG raw tuple/list segment: {seg!r}")
            start, end, label = float(seg[0]), float(seg[1]), seg[2]
        elif isinstance(seg, dict):
            label = seg.get("speaker") or seg.get("label") or "SPEAKER_00"
            start = float(seg.get("start", 0.0))
            end   = float(seg.get("end",   0.0))
        else:
            print(f"[diarization] unrecognized segment type: {type(seg)} -> {seg!r}")
            continue
        # ------------------------------------------------------------------

        if end <= start:
            continue

        parsed.append({"speaker": label, "start": start, "end": end})

    parsed.sort(key=lambda s: s["start"])
    resolved = resolve_overlaps(parsed)

    cleaned = []
    for seg in resolved:
        if cleaned and cleaned[-1]["speaker"] == seg["speaker"]:
            gap = seg["start"] - cleaned[-1]["end"]
            if gap < SAME_SPEAKER_MERGE_GAP:
                cleaned[-1]["end"] = max(cleaned[-1]["end"], seg["end"])
                continue

        cleaned.append({"speaker": seg["speaker"], "start": seg["start"], "end": seg["end"]})

    # Short-segment filtering (< 0.3s)
    filtered = [s for s in cleaned if (s["end"] - s["start"]) >= 0.3]
    return filtered