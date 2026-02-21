"""LLM prompt templates for behavioral UML diagram generation.

Each prompt instructs the LLM to output ONLY PlantUML syntax.
Follows the same role-based pattern as migration/prompts.py.
"""

DIAGRAM_ROLE = """You are an expert software architect who creates precise UML diagrams.
You analyze code structure, call chains, and architectural patterns to produce
accurate PlantUML diagrams. You output ONLY valid PlantUML syntax between
@startuml and @enduml markers — no explanation, no markdown, no commentary."""


def sequence_diagram_prompt(
    mvp_name: str,
    signatures: str,
    call_paths: str,
    source_snippets: str,
) -> str:
    """Build prompt for a Sequence Diagram showing call chain flow."""
    return f"""{DIAGRAM_ROLE}

Generate a PlantUML **sequence diagram** for the MVP "{mvp_name}".

## Unit Signatures
{signatures}

## Call Paths (caller → callee)
{call_paths}

## Source Code Context
{source_snippets}

## Requirements
- Show the primary request flow from entry point through service layers
- Include method calls with meaningful parameter hints
- Show return values where the code makes them clear
- Use `activate`/`deactivate` for method execution spans
- Group related interactions with `alt`, `loop`, or `opt` fragments where appropriate
- Keep participant names short but recognizable (class names, not full qualified names)
- Limit to the most important 15-20 interactions for readability

## Example Format
@startuml
participant "AuthController" as AC
participant "AuthService" as AS
participant "UserRepository" as UR

AC -> AS: authenticate(credentials)
activate AS
AS -> UR: findByUsername(name)
activate UR
UR --> AS: User
deactivate UR
AS --> AC: AuthToken
deactivate AS
@enduml

Now generate the sequence diagram. Output ONLY the PlantUML code:"""


def usecase_diagram_prompt(
    mvp_name: str,
    signatures: str,
    functional_context: str,
) -> str:
    """Build prompt for a Use Case Diagram showing actors and use cases."""
    return f"""{DIAGRAM_ROLE}

Generate a PlantUML **use case diagram** for the MVP "{mvp_name}".

## Unit Signatures
{signatures}

## Functional Context
{functional_context}

## Requirements
- Identify actors (users, external systems, schedulers) from entry points and parameters
- Extract use cases from public methods and API endpoints
- Show include/extend relationships between use cases where appropriate
- Group related use cases in a rectangle (system boundary) labeled with the MVP name
- Keep use case names as short verb phrases ("Log In", "Process Payment")
- Limit to the most significant 8-12 use cases

## Example Format
@startuml
left to right direction
actor "User" as U
actor "Admin" as A

rectangle "Auth Service" {{
  usecase "Log In" as UC1
  usecase "Register" as UC2
  usecase "Reset Password" as UC3
  usecase "Manage Users" as UC4

  UC3 .> UC1 : <<extend>>
}}

U --> UC1
U --> UC2
U --> UC3
A --> UC4
@enduml

Now generate the use case diagram. Output ONLY the PlantUML code:"""


def activity_diagram_prompt(
    mvp_name: str,
    source_snippets: str,
    call_paths: str,
    raptor_summaries: str = "",
) -> str:
    """Build prompt for an Activity Diagram showing control flow."""
    raptor_section = ""
    if raptor_summaries:
        raptor_section = f"""
## Semantic File Summaries (RAPTOR)
These summaries describe what each file in this MVP actually does.
Use them to understand the real behavioral flow even if call paths are sparse.

{raptor_summaries}
"""

    return f"""{DIAGRAM_ROLE}

Generate a PlantUML **activity diagram** for the MVP "{mvp_name}".
{raptor_section}
## Source Code
{source_snippets}

## Call Paths
{call_paths}

## Requirements
- Show the main workflow from entry to completion based on what the code ACTUALLY does
- Infer the activity flow from the semantic summaries above — they describe real behaviors
- Include decision points (if/else, switch) as diamond nodes
- Show parallel activities with fork/join bars where concurrent processing occurs
- Include error handling paths where significant
- Do NOT use swim lane syntax (|Name|) — it causes syntax errors in activity diagrams
- Instead, prefix the first activity from each class/module with its name, e.g. :AuthService.validate;
- Keep the flow focused on the primary happy path with key alternative paths
- Limit to 15-20 activities for readability
- Do NOT invent generic CRUD flows — base every activity on evidence from the code context above

## Example Format
@startuml
start
:Receive Request;
if (Authenticated?) then (yes)
  :Validate Input;
  if (Valid?) then (yes)
    :Process Order;
    :Send Confirmation;
  else (no)
    :Return Validation Error;
  endif
else (no)
  :Return 401;
endif
stop
@enduml

Now generate the activity diagram. Output ONLY the PlantUML code:"""


def deployment_diagram_prompt(
    mvp_name: str,
    target_stack: str,
    architecture_context: str,
    detected_infra: str = "",
) -> str:
    """Build prompt for a Deployment Diagram showing runtime topology.

    Args:
        mvp_name: MVP name
        target_stack: Formatted target stack info
        architecture_context: Architecture context (signatures + call paths)
        detected_infra: Detected frameworks, databases, and external services from ASG
    """
    infra_section = ""
    if detected_infra:
        infra_section = f"""
## Detected Infrastructure (from actual code imports and annotations)
{detected_infra}

IMPORTANT: Base your deployment diagram ONLY on the infrastructure components
listed above. Do NOT invent generic components. Every node in the diagram must
correspond to something detected in the actual code.
"""

    return f"""{DIAGRAM_ROLE}

Generate a PlantUML **deployment diagram** for the MVP "{mvp_name}" in the target architecture.

## Target Stack
{target_stack}

## Architecture Context
{architecture_context}
{infra_section}
## Requirements
- Show deployment nodes (servers, containers, cloud services) as nested rectangles
- Place software artifacts (services, databases, message queues) within their deployment nodes
- Show communication protocols on connections (HTTP, gRPC, AMQP, TCP)
- Include ONLY infrastructure components that are detected in the actual code imports
- Use appropriate stereotypes: <<cloud>>, <<container>>, <<database>>, <<service>>
- Keep focused on the components relevant to this specific MVP
- Do NOT add generic components like "Load Balancer" unless evidence exists in the code

## Example Format
@startuml
node "Load Balancer" as LB <<cloud>>

node "App Server" as APP {{
  artifact "Auth Service" as AUTH <<service>>
  artifact "User API" as API <<service>>
}}

node "Data Layer" {{
  database "PostgreSQL" as DB <<database>>
  database "Redis" as CACHE <<cache>>
}}

LB --> APP : HTTPS
AUTH --> DB : TCP/5432
AUTH --> CACHE : TCP/6379
API --> AUTH : internal
@enduml

Now generate the deployment diagram. Output ONLY the PlantUML code:"""
