import json
import sys
import io
import os

notebook_path = 'c:/Users/Daksh/Downloads/6ab10eb3b23ba_student_resource/notebooks/09_Postprocessing_and_Threshold.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

global_env = {}

for cell_idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        code = "".join(cell.get('source', []))
        
        print(f"Executing cell {cell_idx}...")
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            exec(code, global_env)
            output_text = buffer.getvalue()
            execution_status = "ok"
        except Exception as e:
            output_text = buffer.getvalue() + f"\nException: {str(e)}"
            execution_status = "error"
        finally:
            sys.stdout = old_stdout
            
        print(f"Cell {cell_idx} finished. Status: {execution_status}")
            
        cell['execution_count'] = cell_idx + 1
        cell['outputs'] = []
        if output_text:
            lines = [line + '\n' for line in output_text.split('\n')]
            if lines and lines[-1] == '\n':
                lines = lines[:-1]
            cell['outputs'].append({
                "name": "stdout",
                "output_type": "stream",
                "text": lines
            })
            
        if execution_status == "error":
            print(f"Error in cell {cell_idx}:\n{output_text}")
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook outputs populated successfully!")
