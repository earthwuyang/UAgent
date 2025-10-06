from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import FINISH_TOOL_NAME

_FINISH_DESCRIPTION = """Signals the completion of the ENTIRE task or conversation.

IMPORTANT: DO NOT use this tool after completing intermediate steps or partial work!

ONLY use this tool when:
- The user has EXPLICITLY confirmed they are satisfied and want to end the conversation
- The user says "done", "exit", "quit", or similar farewell
- You have completed ALL parts of a multi-stage task AND the user has no follow-up questions
- You cannot proceed further due to CRITICAL technical limitations (not just waiting for user input)

DO NOT use this tool when:
- You've just completed the first part of a multi-step task
- The user might want to continue, extend, or iterate on the work
- You're waiting for user confirmation or feedback
- You're presenting intermediate results

Instead, use MessageAction to:
- Present your results and ask if the user wants to continue
- Suggest next steps or improvements
- Wait for user confirmation before finishing

The message should include:
- A clear summary of ALL actions taken and their results
- Final status of the COMPLETE task
- Confirmation that the user is satisfied
"""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=FINISH_TOOL_NAME,
        description=_FINISH_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['message'],
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'Final message to send to the user',
                },
            },
        },
    ),
)
