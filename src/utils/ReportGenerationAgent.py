import re
from typing import Any, List
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Context, step
from llama_index.core.workflow import Event
from utils.report_utilities import parse_outline_and_generate_queries
from utils.report_utilities import extract_title

class ReportGenerationEvent(Event):
    pass

class ReportGenerationAgent(Workflow):
    """Report generation agent."""

    def __init__(
        self,
        query_engine: Any,
        llm: FunctionCallingLLM | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.query_engine = query_engine
        self.llm = llm 

    def format_report(self, section_contents, outline):
        """Format the report based on the section contents."""
        report = ""

        introduction_num = "1.1"
        introduction_title = "Introduction"
        conclusion_num = "99.1"
        conclusion_title = "Conclusion"

        for section, subsections in section_contents.items():
            section_match = re.match(r'^(\d+\.)\s*(.*)$', section)
            if section_match:
                section_num, section_title = section_match.groups()
                
                if "introduction" in section.lower():
                    introduction_num, introduction_title = section_num, section_title
                elif "conclusion" in section.lower():
                    conclusion_num, conclusion_title = section_num, section_title
                else:
                    combined_content = "\n".join(subsections.values())
                    summary_query = f"Provide a short summary for section '{section}':\n\n{combined_content}"
                    try:
                        section_summary = str(self.llm.complete(summary_query))
                    except Exception as e:
                        section_summary = f"Error summarizing section: {e}"
                    report += f"# {section_num} {section_title}\n\n{section_summary}\n\n"

                    report = self.get_subsections_content(subsections, report)

        # Add introduction
        introduction_query = f"Create an introduction for the report:\n\n{report}"
        try:
            introduction = str(self.llm.complete(introduction_query))
        except Exception as e:
            introduction = f"Error generating introduction: {e}"
        report = f"# {introduction_num} {introduction_title}\n\n{introduction}\n\n" + report

        # Add conclusion
        conclusion_query = f"Create a conclusion for the report:\n\n{report}"
        try:
            conclusion = str(self.llm.complete(conclusion_query))
        except Exception as e:
            conclusion = f"Error generating conclusion: {e}"
        report += f"# {conclusion_num} {conclusion_title}\n\n{conclusion}"

        # Add title
        title = extract_title(outline)
        report = f"# {title}\n\n{report}"
        return report

    def get_subsections_content(self, subsections, report):
        """Generate content for each subsection in the outline."""
        # Sort subsections by their keys before adding them to the report
        for subsection in sorted(subsections.keys(), key=lambda x: re.search(r'(\d+\.\d+)', x).group(1) if re.search(r'(\d+\.\d+)', x) else x):
            content = subsections[subsection]
            subsection_match = re.search(r'(\d+\.\d+)\.\s*(.+)', subsection)
            if subsection_match:
                subsection_num, subsection_title = subsection_match.groups()
                report += f"## {subsection_num} {subsection_title}\n\n{content}\n\n"
            else:
                report += f"## {subsection}\n\n{content}\n\n"
        return report

    def generate_section_content(self, queries, reverse=False):
        """Generate content for each section and subsection in the outline."""
        section_contents = {}
        for section, subsections in queries.items():
            section_contents[section] = {}
            subsection_keys = reversed(sorted(subsections.keys())) if reverse else sorted(subsections.keys())
            for subsection in subsection_keys:
                data = subsections[subsection]
                query = data['query']
                classification = data['classification']
                try:
                    if classification == "LLM":
                        answer = str(self.llm.complete(query + " Give a short answer."))
                    else:
                        answer = str(self.query_engine.query(query))
                except Exception as e:
                    answer = f"Error generating answer for {subsection}: {e}"
                section_contents[section][subsection] = answer
        return section_contents

    @step(pass_context=True)
    async def queries_generation_event(self, ctx: Context, ev: StartEvent) -> ReportGenerationEvent:
        """Generate queries for the report."""
        ctx.data["outline"] = ev.outline
        queries = parse_outline_and_generate_queries(llm=self.llm,outline=ctx.data["outline"])

        return ReportGenerationEvent(queries=queries)


    @step(pass_context=True)
    async def generate_report(
        self, ctx: Context, ev: ReportGenerationEvent
    ) -> StopEvent:
        """Generate report."""
        queries = ev.queries

        # Generate contents for sections in reverse order
        section_contents = self.generate_section_content(queries, reverse=True)
        # Format and compile the final report
        report = self.format_report(section_contents, ctx.data["outline"])

        return StopEvent(result={"response": report})
