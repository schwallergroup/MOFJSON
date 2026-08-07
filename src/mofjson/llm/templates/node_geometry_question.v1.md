---
id: node_geometry_question
version: 1
variables: [metal_node_json, question]
purpose: Answer a chemical question about a MOFJSON metal_node document.
---
Given the following MOFJSON metal node, answer the question.
Return JSON only matching the provided answer schema.

Metal node:
{metal_node_json}

Question:
{question}
