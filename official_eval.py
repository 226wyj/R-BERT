import os

# platform_ = platform.system()
# if platform_ == "Windows":
#     EVAL_DIR = '.\\DataSets\\SemEval\\eval'
# elif platform_ == "Linux":
EVAL_DIR = './DataSets/SemEval/eval'

# Run the perl script
try:
    cmd = r"perl {0}/semeval2010_task8_scorer-v1.2.pl {0}/predicted_answers.txt {0}/answer_keys.txt > {0}/result.txt".format(EVAL_DIR)
    os.system(cmd)
except:
    raise Exception("perl is not installed or proposed_answers.txt is missing")

with open(os.path.join(EVAL_DIR, 'result.txt'), 'r', encoding='utf-8') as f:
    macro_result = list(f)[-1]
    macro_result = macro_result.split(":")[1].replace(">>>", "").strip()
    print(macro_result)
