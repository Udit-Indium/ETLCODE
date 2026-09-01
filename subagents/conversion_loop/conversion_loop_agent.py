from google.adk.agents import LoopAgent, SequentialAgent

from .code_converter import code_convertor_agent
from .case_fact_validation_agent import case_fact_checker_agent

code_convertor_loop_agent = LoopAgent(
    name="code_convertor_loop_agent",
    description=(
        "Converts the Python script to PySpark and fact-checks the result, "
        "looping until the case facts match or max_iterations is reached."
    ),
    sub_agents=[code_convertor_agent, case_fact_checker_agent],
    # Budget = ceil(functions / BATCH_SIZE) + headroom for runtime-error fix
    # passes. BATCH_SIZE is 8 (see code_convertor_agent), and the converter
    # stops its turn after each batch, so one iteration converts AT MOST 8
    # functions — fewer when a batch does not fit in its output limit.
    #
    # The previous comment assumed ~20 per iteration and set this to 8, which
    # capped the loop at 64 functions. The shortfall is silent: the loop simply
    # exhausts its iterations with functions still unconverted, and semantic
    # validation then runs against an incomplete module.
    #
    # 16 covers ~128 functions, comfortably clear of the ~19 the refactor now
    # produces for the current source. If your source is much larger, raise
    # this or BATCH_SIZE.
    max_iterations=16,
)

# documentation_generator_agent used to be the second stage here. It now runs at
# the END of the orchestrator, after semantic validation, for two reasons: the
# semantic code fixer edits the converted module, so documenting it here
# described code that was about to change; and the document's semantic-validation
# section needs a verdict that did not exist yet at this point in the run.
#
# The wrapper is left in place with a single child: it is the name the
# orchestrator imports, and it is where a later post-conversion stage would go.
conversion_loop_agent = SequentialAgent(
    name="conversion_loop_agent",
    description="A sequential agent that orchestrates the code conversion and fact-checking process.",
    sub_agents=[code_convertor_loop_agent]
)
