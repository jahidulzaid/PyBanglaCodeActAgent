import json
import os
import ast
import pandas as pd
import time
import signal

reference_dir = "" #replace with the path to the reference data (dev.csv)
prediction_dir = "" #replace with the path to the prediction data (submission.json)


#Do not modify anything below this part

# Timeout handler
def handler(signum, frame):
    raise TimeoutError("Execution timed out")


def evaluate_combined_data(res_data, ref_data):
    # Convert to DataFrames for easy merging
    res_df = pd.DataFrame(res_data)[['id', 'response']]
    ref_df = pd.DataFrame(ref_data)
    # Drop the response column from ref_df if it exists
    if 'response' in ref_df.columns:
        ref_df = ref_df.drop(columns=['response'])
    
    # Merge the data on 'id'
    combined_df = ref_df.merge(res_df, on='id', how='left')
    
    # Convert back to list of dictionaries
    combined_data = combined_df.to_dict('records')
    
    global_correct = 0
    global_total = len(combined_data)
    
    for entry in combined_data:
        entry_id = entry['id']
        response_code = entry.get('response', '')  # Use empty string if response missing
        test_list_raw = entry['test_list']
        if response_code is not None:
            response_code = response_code.strip('` \n').replace('python\n', '').strip()
        
        
        print(f"Executing Sample ID: {entry_id}")
        
        # 🚫 Skip code if it contains time.sleep (case-insensitive)
        if "time.sleep" in response_code.lower():
            print(f"⏭️ Skipping Code Execution: contains time.sleep()")
            continue

        correct = 0
        

        # Parse the test cases safely
        try:
            inner_str = ast.literal_eval(test_list_raw)
            test_cases = ast.literal_eval(inner_str)
        except Exception as e:
            print(f"❌❌❌❌ Failed to parse test_list: {e} ❌❌❌")
            continue

        # Create a shared namespace for exec
        namespace = {}

        try:
            # Set timeout for function definition
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(30)
            exec(response_code, namespace)
            signal.alarm(0)  # cancel timer if finished early
        except TimeoutError:
            print(f"⏱️ Timeout in function definition. Skipping test case execution for this ID.\n")
            continue
        except Exception as e:
            print(f"❌ Error in function definition: {e}. Skipping test case execution for this ID.\n")
            continue

        passed = True
        # Run each assert statement
        for i, assert_stmt in enumerate(test_cases):
            try:
                signal.alarm(30)  # 30 seconds per test case
                exec(assert_stmt, namespace)
                signal.alarm(0)
                correct += 1
            except TimeoutError:
                print(f"⏱️ Test case {i + 1} timed out. Skipping all remaining test cases for this ID.")
                passed = False
                break  # Exit loop on timeout
            except AssertionError:
                print(f"❌ Test case {i + 1} failed: assertion error. Skipping all remaining test cases for this ID.")
                passed = False
                break  # Exit loop on timeout
            except Exception as e:
                print(f"⚠️ Test case {i + 1} exception: {e}. Skipping all remaining test cases for this ID.")
                passed = False
                break  # Exit loop on timeout
            finally:
                signal.alarm(0)
        if passed:
            print(f"✅ ID {entry_id} Passed all test cases.\n")
        else:
            print(f"❌ ID {entry_id} Failed some test cases.\n")

        total = len(test_cases)
        if correct == total:
            global_correct += 1
        
    return global_correct, global_total




# Read both files
with open(os.path.join(prediction_dir,'submission.json'), 'r', encoding='utf-8') as f:
    res_data = json.load(f)



ref_df = pd.read_csv(
    os.path.join(reference_dir, 'dev.csv'),
    dtype=str,                # keep everything as string to avoid NaN
    keep_default_na=False     # empty cells stay '', not NaN
)
# Ensure 'id' is numeric to merge cleanly (adjust to int if your JSON ids are ints)
ref_df['id'] = ref_df['id'].astype(int)
ref_data = ref_df.to_dict('records')

# Evaluate the combined data
correct, all = evaluate_combined_data(res_data, ref_data)

# Write the accuracy to scores.json
scores = {
    "accuracy": correct / all if all > 0 else 0.0
}

print(f"\nPass@1: {correct}/{all} = {scores['accuracy']:.2f}")


