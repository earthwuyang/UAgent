from datasets import load_dataset

# Load the MATH-500 test set
MATH_test_set = load_dataset("HuggingFaceH4/MATH-500")["test"]

# Extract the 'problem' and 'solution' fields
problems = MATH_test_set["problem"]
solutions = MATH_test_set["solution"]

# Print the first few problems and solutions to verify
print(problems[:5])
print(solutions[:5])