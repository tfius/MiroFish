"""
Ontology generation service
Interface 1: Analyse text content and generate entity and relationship type definitions
suitable for social simulation
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# System prompt for ontology generation
ONTOLOGY_SYSTEM_PROMPT = """You are a professional knowledge graph ontology design expert. Your task is to analyse the given text and simulation requirements and design entity types and relationship types suitable for **social media public-opinion simulation**.

**Important: you must output valid JSON only — do not output anything else.**

## Core Task Background

We are building a **social media public-opinion simulation system** in which:
- Every entity is an "account" or "actor" that can speak, interact, and spread information on social media
- Entities influence one another, retweet, comment, and respond
- We need to simulate each party's reactions and information propagation paths in a public-opinion event

Therefore, **entities must be real-world actors that can speak and interact on social media**:

**Allowed**:
- Specific individuals (public figures, parties involved, opinion leaders, experts and scholars, ordinary people)
- Companies and enterprises (including their official accounts)
- Organisations (universities, associations, NGOs, trade unions, etc.)
- Government departments and regulatory bodies
- Media organisations (newspapers, TV stations, independent media, websites)
- Social media platforms themselves
- Representatives of specific groups (e.g. alumni associations, fan clubs, advocacy groups)

**Not allowed**:
- Abstract concepts (e.g. "public opinion", "sentiment", "trends")
- Topics/subjects (e.g. "academic integrity", "education reform")
- Viewpoints/stances (e.g. "supporters", "opponents")

## Output Format

Output JSON with the following structure:

```json
{
    "entity_types": [
        {
            "name": "Entity type name (English, PascalCase)",
            "description": "Short description (English, max 100 characters)",
            "attributes": [
                {
                    "name": "attribute name (English, snake_case)",
                    "type": "text",
                    "description": "Attribute description"
                }
            ],
            "examples": ["Example entity 1", "Example entity 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Relationship type name (English, UPPER_SNAKE_CASE)",
            "description": "Short description (English, max 100 characters)",
            "source_targets": [
                {"source": "Source entity type", "target": "Target entity type"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brief analysis summary of the text content (Chinese)"
}
```

## Design Guidelines (extremely important!)

### 1. Entity Type Design — must be followed strictly

**Quantity requirement: exactly 10 entity types**

**Hierarchy requirement (must include both concrete types and fallback types)**:

Your 10 entity types must contain the following levels:

A. **Fallback types (mandatory, placed as the last 2 items in the list)**:
   - `Person`: Fallback type for any individual natural person. Use this when a person does not fit any more specific person type.
   - `Organization`: Fallback type for any organisation. Use this when an organisation does not fit any more specific organisation type.

B. **Concrete types (8 types, designed based on the text content)**:
   - Design more specific types for the main roles appearing in the text
   - Example: if the text involves an academic event, types might include `Student`, `Professor`, `University`
   - Example: if the text involves a business event, types might include `Company`, `CEO`, `Employee`

**Why fallback types are needed**:
- The text will contain many kinds of people, such as "primary school teachers", "passers-by", and "anonymous netizens"
- If there is no specific type to match, they should fall back to `Person`
- Similarly, small organisations and temporary groups should fall back to `Organization`

**Principles for designing concrete types**:
- Identify the most frequently occurring or key role types in the text
- Each concrete type should have a clear boundary to avoid overlap
- The description must clearly explain how this type differs from the fallback types

### 2. Relationship Type Design

- Quantity: 6–10 types
- Relationships should reflect real connections found in social media interactions
- Ensure that source_targets cover the entity types you have defined

### 3. Attribute Design

- 1–3 key attributes per entity type
- **Note**: attribute names must not use `name`, `uuid`, `group_id`, `created_at`, or `summary` (these are system-reserved words)
- Recommended names: `full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Entity Type Reference

**Individual (concrete)**:
- Student: student
- Professor: professor/scholar
- Journalist: journalist
- Celebrity: celebrity/influencer
- Executive: executive
- Official: government official
- Lawyer: lawyer
- Doctor: doctor

**Individual (fallback)**:
- Person: any natural person (used when none of the concrete types apply)

**Organisation (concrete)**:
- University: university or college
- Company: company or enterprise
- GovernmentAgency: government agency
- MediaOutlet: media organisation
- Hospital: hospital
- School: primary or secondary school
- NGO: non-governmental organisation

**Organisation (fallback)**:
- Organization: any organisation (used when none of the concrete types apply)

## Relationship Type Reference

- WORKS_FOR: works for
- STUDIES_AT: studies at
- AFFILIATED_WITH: affiliated with
- REPRESENTS: represents
- REGULATES: regulates
- REPORTS_ON: reports on
- COMMENTS_ON: comments on
- RESPONDS_TO: responds to
- SUPPORTS: supports
- OPPOSES: opposes
- COLLABORATES_WITH: collaborates with
- COMPETES_WITH: competes with
"""


class OntologyGenerator:
    """
    Ontology generator
    Analyses text content and generates entity and relationship type definitions
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate the ontology definition

        Args:
            document_texts: List of document texts
            simulation_requirement: Simulation requirement description
            additional_context: Additional context

        Returns:
            Ontology definition (entity_types, edge_types, etc.)
        """
        # Build user message
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # Call the LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=64000 # was 16000
        )
        
        # Validate and post-process
        result = self._validate_and_process(result)
        
        return result
    
    # Maximum text length passed to the LLM (50,000 characters)
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build the user message"""

        # Merge texts
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # If the text exceeds 50,000 characters, truncate it
        # (only affects content sent to the LLM; does not affect graph construction)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Original text is {original_length} characters; first {self.MAX_TEXT_LENGTH_FOR_LLM} characters used for ontology analysis)..."

        message = f"""## Simulation Requirement

{simulation_requirement}

## Document Content

{combined_text}
"""

        if additional_context:
            message += f"""
## Additional Notes

{additional_context}
"""

        message += """
Based on the above content, design entity types and relationship types suitable for social public-opinion simulation.

**Rules that must be followed**:
1. You must output exactly 10 entity types
2. The last 2 must be fallback types: Person (fallback for individuals) and Organization (fallback for organisations)
3. The first 8 are concrete types designed based on the text content
4. All entity types must be real-world actors capable of speaking publicly; they cannot be abstract concepts
5. Attribute names must not use reserved words such as name, uuid, group_id; use full_name, org_name, etc. instead
"""

        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process the result"""

        # Ensure required fields exist
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Validate entity types
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Ensure description does not exceed 100 characters
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Validate relationship types
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Zep API limits: max 10 custom entity types, max 10 custom edge types
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # Fallback type definitions
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # Check whether fallback types are already present
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # Collect fallback types that need to be added
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # If adding them would exceed 10, remove some existing types
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calculate how many to remove
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Remove from the end (preserve the more important concrete types at the front)
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Add fallback types
            result["entity_types"].extend(fallbacks_to_add)

        # Final guard to ensure limits are not exceeded (defensive programming)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Convert an ontology definition to Python code (similar to ontology.py)

        Args:
            ontology: Ontology definition

        Returns:
            Python code as a string
        """
        code_lines = [
            '"""',
            'Custom entity type definitions',
            'Auto-generated by MiroFish for social public-opinion simulation',
            '"""',
            '',
            'from pydantic import Field',
            'from app.local_zep.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entity Type Definitions ==============',
            '',
        ]

        # Generate entity types
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== Relationship Type Definitions ==============')
        code_lines.append('')

        # Generate relationship types
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Convert to PascalCase class name
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # Generate type dictionaries
        code_lines.append('# ============== Type Configuration ==============')

        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # Generate edge source_targets mapping
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

