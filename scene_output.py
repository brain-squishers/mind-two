def build_scene_output_payload(
    user_query,
    target_label,
    spatial_index,
    relations,
    max_relations=3,
):
    target_entry = spatial_index.get("target_1")
    if target_entry is None:
        return None

    filtered_relations = []
    for relation in relations[:max_relations]:
        filtered_relations.append(
            {
                "type": relation["type"],
                "object": relation["object"],
                "score": round(float(relation["score"]), 3),
            }
        )

    ego_relation = next(
        (
            relation["type"]
            for relation in relations
            if relation["type"] in {"on_your_left", "on_your_right", "straight_ahead"}
        ),
        None,
    )

    return {
        "user_query": user_query,
        "target_label": target_label,
        "target_position_3d_m": target_entry["position_3d_m"],
        "target_depth_m": target_entry["depth_median_m"],
        "target_depth_std_m": target_entry["depth_std_m"],
        "ego_relation": ego_relation,
        "relations": filtered_relations,
    }
