import sys

with open('run_spst.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix unpacking
content = content.replace("output, prov, auth_result, schema_ok, errs = task.execute(", "res = task.execute(")
content = content.replace("c[\"case_id\"], c[\"input\"]\n        )", "c[\"case_id\"], c[\"input\"]\n        )\n        output = res.extracted_output\n        prov = res.provenance\n        auth_result = res.authorization\n        schema_ok = res.schema_valid\n        errs = res.schema_errors")

with open('run_spst.py', 'w', encoding='utf-8') as f:
    f.write(content)
