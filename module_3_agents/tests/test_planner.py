import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import src.agents.config

from src.agents.nodes import planner_node

def test_planner():
    initial_state = {
        "global_topic": "Methods of nanostructuring and surface modification",
        "pending_sections": [],
        "current_section": None,
        "completed_sections": [],
        "final_document": ""
    }

    print(f"Запускаем Планировщик для темы: '{initial_state['global_topic']}'...\n")
    print("=" * 60)

    try:
        new_state = planner_node(initial_state)

        print("\n=== ПЛАН УСПЕШНО СГЕНЕРИРОВАН ===\n")

        current = new_state.get("current_section")
        if current:
            print("▶ ПЕРВАЯ СЕКЦИЯ (current_section):")
            print(f"  ID: {current['section_id']}")
            print(f"  Название: {current['title']}")
            print("  Пути в базе (target_paths):")
            for path in current['target_paths']:
                print(f"    - {path}")
            print(f"  Инструкция:\n    {current['instructions']}\n")

        pending = new_state.get("pending_sections", [])
        print("=" * 60)
        print(f"▶ ОЧЕРЕДЬ СЕКЦИЙ (pending_sections: {len(pending)}):")
        print("=" * 60)

        for sec in pending:
            print(f"  ID: {sec['section_id']} | {sec['title']}")
            print("  Пути в базе (target_paths):")
            if sec['target_paths']:
                for path in sec['target_paths']:
                    print(f"    - {path}")
            else:
                print("    - (пусто)")
            print(f"  Инструкция:\n    {sec['instructions']}\n")
            print("-" * 40)

    except Exception as e:
        print(f"\nОШИБКА ПРИ ВЫПОЛНЕНИИ: {e}")

if __name__ == "__main__":
    test_planner()