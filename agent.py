import asyncio
import os
from typing import Any

import dotenv
from github import Github
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult, ToolCall, FunctionAgent, AgentWorkflow
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context

dotenv.load_dotenv()

full_repo_name = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else Github()

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)


def get_repo():
    return git.get_repo(full_repo_name)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Fetch PR details by number: author, title, body, diff URL, state, head SHA and commit SHAs."""
    pull_request = get_repo().get_pull(pr_number)
    commit_SHAs = [c.sha for c in pull_request.get_commits()]
    return {
        "user": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "commit_SHAs": commit_SHAs,
    }


def get_file_contents(file_path: str) -> str:
    """Fetch the contents of a file from the repository by its path."""
    return get_repo().get_contents(file_path).decoded_content.decode('utf-8')


def get_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Fetch the files changed in a commit (by SHA) with their status, line counts and patch."""
    commit = get_repo().get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Save the gathered PR context (details, changed files, requested files) to the shared state."""
    state = await ctx.store.get("state")
    state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", state)
    return "State updated with gathered contexts."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Save the draft PR review comment to state."""
    state = await ctx.store.get("state")
    state["draft_comment"] = draft_comment
    await ctx.store.set("state", state)
    return "State updated with draft comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Save the final, checked PR review to state."""
    state = await ctx.store.get("state")
    state["final_review"] = final_review
    await ctx.store.set("state", state)
    return "State updated with final review."


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Post the final review comment to the GitHub pull request"""
    pull_request = get_repo().get_pull(pr_number)
    pull_request.create_review(body=comment,event="COMMENT")
    return "Review posted to Github."


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
commit_details_tool = FunctionTool.from_defaults(get_commit_details)
add_context_to_state_tool = FunctionTool.from_defaults(add_context_to_state)
add_comment_to_state_tool = FunctionTool.from_defaults(add_comment_to_state)
add_final_review_to_state_tool = FunctionTool.from_defaults(add_final_review_to_state)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)

context_agent = FunctionAgent(
    name="ContextAgent",
    description="Gathers all the needed context about a pull request: PR details, changed files, and any requested repository files.",
    llm=llm,
    tools=[pr_details_tool, file_contents_tool, commit_details_tool, add_context_to_state_tool],
    can_handoff_to = ["CommentorAgent"],
    system_prompt="""You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. """,
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[add_comment_to_state_tool],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
    system_prompt="""You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the ContextAgent. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
 - IMPORTANT: Do NOT return the review as your final answer. When the review is written, you MUST:
    1. Save it with the add_comment_to_state tool;
    2. Then hand off to the ReviewAndPostingAgent for the final check and posting. \n
 """
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft PR comment, requests rewrites if needed, and posts the final review to GitHub.",
    tools=[add_final_review_to_state_tool, post_review_tool],
    can_handoff_to=["CommentorAgent"],
    system_prompt="""
    You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
    Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
    If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
    When you are satisfied, post the review to GitHub.  
    """
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={"gathered_contexts": "", "draft_comment": "", "final_review": ""},
)


async def main():
    query = f"Write a review for PR number {pr_number} and post it to GitHub."
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
