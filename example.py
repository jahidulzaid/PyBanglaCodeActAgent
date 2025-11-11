"""
Example usage of PyBanglaCodeActAgent agent.

This script demonstrates how to use the PyBanglaCodeActAgent agent
for a simple programming task.
"""

from PyBanglaCodeAct import (
    initialize_llm,
    CodeActAgent,
    llm_engine,
)


def main():
    """Run a simple example."""
    print("🚀 Initializing PyBanglaCodeActAgent...")
    
    # Initialize the LLM
    initialize_llm("Qwen/Qwen3-8B")
    
    # Create an agent
    agent = CodeActAgent(
        llm_engine=llm_engine,
        max_iterations=4,
    )
    
    # Example task in Bangla
    task = """You are solving a coding task.
Instruction: একটি ফাংশন লিখুন যা দুটি সংখ্যার যোগফল রিটার্ন করবে।

Here are the test cases you must satisfy:
assert add(2, 3) == 5
assert add(-1, 1) == 0
assert add(0, 0) == 0

Please return only the Python function/code solution, nothing else.
"""
    
    print("\n📝 Task:")
    print(task)
    print("\n" + "="*50)
    
    # Run the agent
    response = agent.run(task)
    
    print("\n" + "="*50)
    print("\n✅ Generated Solution:")
    print(response)


if __name__ == "__main__":
    main()
