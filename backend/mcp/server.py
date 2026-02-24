"""
MCP Server — Model Context Protocol server for tool registration,
structured outputs, and controlled system prompts.
"""

import logging
import json
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Model Context Protocol server that manages:
    - Tool registration and execution
    - Structured output schemas
    - System prompt management
    """

    def __init__(self):
        self.tools = {}
        self.schemas = {}
        self.system_prompts = {}
        logger.info("MCP Server initialized")

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
    ):
        """Register a tool with the MCP server."""
        self.tools[name] = {
            "handler": handler,
            "description": description,
            "input_schema": input_schema or {},
            "output_schema": output_schema or {},
        }
        if output_schema:
            self.schemas[name] = output_schema
        logger.info(f"Tool registered: {name}")

    def register_system_prompt(self, agent_role: str, prompt: str):
        """Register a system prompt for an agent role."""
        self.system_prompts[agent_role] = prompt
        logger.info(f"System prompt registered for role: {agent_role}")

    def execute_tool(self, name: str, **kwargs) -> dict:
        """Execute a registered tool."""
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}

        tool = self.tools[name]
        try:
            result = tool["handler"](**kwargs)

            # Validate output against schema if available
            if name in self.schemas:
                result = self._validate_output(result, self.schemas[name])

            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Tool execution failed: {name} - {e}")
            return {"success": False, "error": str(e)}

    def get_system_prompt(self, agent_role: str) -> str:
        """Get the system prompt for an agent role."""
        return self.system_prompts.get(agent_role, "You are a helpful assistant.")

    def list_tools(self) -> list:
        """List all registered tools."""
        return [
            {
                "name": name,
                "description": tool["description"],
                "input_schema": tool["input_schema"],
            }
            for name, tool in self.tools.items()
        ]

    def _validate_output(self, output: Any, schema: dict) -> Any:
        """Basic schema validation for structured outputs."""
        if isinstance(output, dict) and "required_fields" in schema:
            for field in schema["required_fields"]:
                if field not in output:
                    output[field] = schema.get("defaults", {}).get(field, None)
        return output


# Global MCP server instance
_mcp_server = None


def get_mcp_server() -> MCPServer:
    """Get or create the global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
        _register_default_tools(_mcp_server)
        _register_system_prompts(_mcp_server)
    return _mcp_server


def _register_default_tools(server: MCPServer):
    """Register default interview tools."""

    # Resume parser tool
    from agents.resume_analyzer import analyze_resume
    server.register_tool(
        name="resume_parser",
        handler=lambda pdf_bytes: analyze_resume(pdf_bytes),
        description="Parse a PDF resume and extract structured candidate profile",
        output_schema={
            "required_fields": ["name", "skills", "domains", "experience_years"],
            "defaults": {"name": "Unknown", "skills": [], "domains": [], "experience_years": 0},
        }
    )

    # Question retriever tool
    from rag.retriever import retrieve_questions
    server.register_tool(
        name="db_retriever",
        handler=lambda query, domain=None, difficulty=None, top_k=5:
            retrieve_questions(query, domain, difficulty, top_k),
        description="Retrieve relevant interview questions from the knowledge base",
        output_schema={
            "required_fields": [],
        }
    )

    # Scoring tool
    from agents.response_analyzer import analyze_response
    server.register_tool(
        name="scoring_module",
        handler=lambda question, answer, domain="general", difficulty="medium":
            analyze_response(question, answer, domain, difficulty),
        description="Score a candidate's answer and detect AI-generated content",
        output_schema={
            "required_fields": ["score", "feedback", "ai_probability"],
            "defaults": {"score": 0, "feedback": "", "ai_probability": 0},
        }
    )

    logger.info("Default tools registered with MCP server")


def _register_system_prompts(server: MCPServer):
    """Register system prompts for each agent role."""

    server.register_system_prompt(
        "resume_analyzer",
        "You are an expert HR professional analyzing candidate resumes. "
        "Extract skills, experience, and qualifications accurately. "
        "Always return structured JSON output."
    )

    server.register_system_prompt(
        "interview_planner",
        "You are a senior technical interviewer planning an interview. "
        "Design adaptive question sequences that match candidate experience. "
        "Balance topic coverage and progressive difficulty."
    )

    server.register_system_prompt(
        "question_generator",
        "You are a technical interviewer generating questions. "
        "Create clear, specific questions that test real understanding. "
        "Avoid generic or overly broad questions. Always respond in JSON format."
    )

    server.register_system_prompt(
        "response_analyzer",
        "You are an objective answer evaluator. Score answers fairly "
        "based on correctness, depth, and practical understanding. "
        "Detect potential AI-generated content. Always respond in JSON format."
    )

    server.register_system_prompt(
        "report_generator",
        "You are an HR assessment specialist generating interview reports. "
        "Provide balanced, evidence-based recommendations. "
        "Include both strengths and areas for improvement."
    )
