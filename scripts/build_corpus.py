"""
Corpus builder for ThreatMatch.
Parses STIX 2.1 Enterprise ATT&CK JSON and extracts:
- All 800+ Techniques and Sub-techniques (id, name, description, tactics, platforms, mitigations, url)
  with backward-compatible support for historical and current IDs.
- Mitigations (id, name, description)
- Tactics (id, name, shortname, description)
Saves structured documents to data/mitre_attack_corpus.json.
"""

import json
import os
import sys

def build_corpus(raw_path="data/enterprise_attack_raw.json", output_path="data/mitre_attack_corpus.json"):
    print(f"Loading raw STIX data from {raw_path}...")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw STIX file not found at {raw_path}. Please download it first.")

    with open(raw_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects = bundle.get("objects", [])
    print(f"Total STIX objects found: {len(objects)}")

    # 1. Parse Mitigations (course-of-action)
    coas = {}
    for obj in objects:
        if obj.get("type") == "course-of-action":
            ext_refs = obj.get("external_references", [])
            mitre_ref = next((r for r in ext_refs if r.get("source_name") == "mitre-attack"), None)
            mid = mitre_ref.get("external_id") if mitre_ref else obj.get("id")
            coas[obj.get("id")] = {
                "id": mid,
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "is_deprecated": obj.get("x_mitre_deprecated", False),
                "is_revoked": obj.get("revoked", False)
            }
    print(f"Extracted {len(coas)} mitigations.")

    # 2. Parse Tactics (x-mitre-tactic)
    tactics_map = {}
    for obj in objects:
        if obj.get("type") == "x-mitre-tactic":
            ext_refs = obj.get("external_references", [])
            mitre_ref = next((r for r in ext_refs if r.get("source_name") == "mitre-attack"), None)
            tid = mitre_ref.get("external_id") if mitre_ref else obj.get("id")
            shortname = obj.get("x_mitre_shortname")
            tactics_map[shortname] = {
                "id": tid,
                "name": obj.get("name", ""),
                "description": obj.get("description", "")
            }
    print(f"Extracted {len(tactics_map)} tactics.")

    # 3. Parse Relationships: mitigates, subtechnique-of, revoked-by
    mitigations_by_technique = {}
    subtechnique_of_map = {}
    revoked_by_map = {}
    for obj in objects:
        if obj.get("type") == "relationship":
            rel_type = obj.get("relationship_type")
            src = obj.get("source_ref")
            tgt = obj.get("target_ref")
            if rel_type == "mitigates" and src in coas:
                mitigations_by_technique.setdefault(tgt, []).append({
                    "id": coas[src]["id"],
                    "name": coas[src]["name"],
                    "description": obj.get("description") or coas[src]["description"]
                })
            elif rel_type == "subtechnique-of":
                subtechnique_of_map[src] = tgt
            elif rel_type == "revoked-by":
                revoked_by_map[src] = tgt

    # 4. Map STIX IDs to external IDs
    stix_id_to_tech_id = {}
    for obj in objects:
        if obj.get("type") == "attack-pattern":
            ext_refs = obj.get("external_references", [])
            mitre_ref = next((r for r in ext_refs if r.get("source_name") == "mitre-attack"), None)
            if mitre_ref and mitre_ref.get("external_id"):
                stix_id_to_tech_id[obj.get("id")] = mitre_ref.get("external_id")

    # 5. Parse Attack Patterns (Techniques and Sub-techniques)
    techniques = []
    for obj in objects:
        if obj.get("type") == "attack-pattern":
            ext_refs = obj.get("external_references", [])
            mitre_ref = next((r for r in ext_refs if r.get("source_name") == "mitre-attack"), None)
            if not mitre_ref or not mitre_ref.get("external_id"):
                continue

            tech_id = mitre_ref.get("external_id")
            name = obj.get("name", "")
            desc = obj.get("description", "")
            detection = obj.get("x_mitre_detection", "")
            platforms = obj.get("x_mitre_platforms", [])
            is_sub = obj.get("x_mitre_is_subtechnique", False)
            is_revoked = obj.get("revoked", False)
            is_deprecated = obj.get("x_mitre_deprecated", False)
            parent_stix = subtechnique_of_map.get(obj.get("id"))
            parent_id = stix_id_to_tech_id.get(parent_stix, "")
            successor_stix = revoked_by_map.get(obj.get("id"))
            successor_id = stix_id_to_tech_id.get(successor_stix, "")

            # Resolve tactics
            tech_tactics = []
            for phase in obj.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") == "mitre-attack":
                    pname = phase.get("phase_name")
                    t_info = tactics_map.get(pname)
                    t_name = t_info["name"] if t_info else pname.replace("-", " ").title()
                    tech_tactics.append(t_name)
            tech_tactics = sorted(list(set(tech_tactics)))

            # Mitigations
            tech_mitigations = mitigations_by_technique.get(obj.get("id"), [])
            seen_m = set()
            unique_mitigations = []
            for m in tech_mitigations:
                if m["id"] not in seen_m:
                    seen_m.add(m["id"])
                    unique_mitigations.append(m)

            # Build unified searchable content
            mitigation_text = " ".join([f"{m['name']}: {m['description']}" for m in unique_mitigations])
            tactic_str = ", ".join(tech_tactics)
            platform_str = ", ".join(platforms)
            successor_str = f" Successor: {successor_id}" if successor_id else ""
            search_text = (
                f"{tech_id} {name}{successor_str}\n"
                f"Tactics: {tactic_str}\n"
                f"Platforms: {platform_str}\n"
                f"Description: {desc}\n"
                f"Detection: {detection}\n"
                f"Mitigations: {mitigation_text}"
            )

            techniques.append({
                "id": tech_id,
                "stix_id": obj.get("id"),
                "name": name,
                "url": mitre_ref.get("url", f"https://attack.mitre.org/techniques/{tech_id.replace('.', '/')}"),
                "tactics": tech_tactics,
                "platforms": platforms,
                "description": desc,
                "detection": detection,
                "is_subtechnique": is_sub,
                "is_revoked": is_revoked,
                "is_deprecated": is_deprecated,
                "parent_id": parent_id,
                "successor_id": successor_id,
                "mitigations": unique_mitigations,
                "search_text": search_text
            })

    # Sort techniques by ID (e.g. T1001, T1001.001, T1002...)
    techniques.sort(key=lambda x: x["id"])

    corpus = {
        "metadata": {
            "version": "MITRE ATT&CK Enterprise STIX 2.1",
            "total_techniques": len(techniques),
            "total_mitigations": len(coas),
            "total_tactics": len(tactics_map)
        },
        "tactics": tactics_map,
        "techniques": techniques
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated corpus with {len(techniques)} techniques saved to {output_path}!")
    return corpus

if __name__ == "__main__":
    build_corpus()
