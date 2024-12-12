import pm4py

from utils.llm_model_generator import LLMProcessModelGenerator
import os
import json
import random
import traceback

model = "Qwen/QwQ-32B-Preview"
api_key = "ZFwz90wwIzRwPcrTGbecxSq6SjOUClrv"
api_url = "https://api.deepinfra.com/v1/openai"
print("== starting")

gt_pro_desc_folder = r"C:\Users\berti\EvaluatingLLMsProcessModeling\ground_truth\ground_truth_process_descriptions\long"
gt_pn_folder = r"C:\Users\berti\EvaluatingLLMsProcessModeling\ground_truth\ground_truth_pn"

fold1_content = os.listdir(gt_pro_desc_folder)
fold2_content = os.listdir(gt_pn_folder)

root_folder = r"C:\Users\berti\EvaluatingLLMsProcessModeling\results_benchmarking\QwQ-32B-Preview"
code_folder = os.path.join(root_folder, "code")
conv_folder = os.path.join(root_folder, "conv")
pn_folder = os.path.join(root_folder, "pn")

if not os.path.exists(code_folder):
    os.mkdir(code_folder)

if not os.path.exists(conv_folder):
    os.mkdir(conv_folder)

if not os.path.exists(pn_folder):
    os.mkdir(pn_folder)

for idx in range(len(fold1_content)):
    print("==", idx)

    target_code = os.path.join(code_folder, str(idx+1).zfill(2)+".txt")
    target_conv = os.path.join(conv_folder, str(idx+1).zfill(2)+".txt")
    target_pn = os.path.join(pn_folder, str(idx+1).zfill(2)+".pnml")

    if not os.path.exists(target_pn):
        try:
            F = open(os.path.join(gt_pro_desc_folder, fold1_content[idx]), "r")
            pro_desc = F.read().strip()
            F.close()

            net0, im0, fm0 = pm4py.read_pnml(os.path.join(gt_pn_folder, fold2_content[idx]))
            activities = [x.label for x in net0.transitions if x.label is not None]
            random.shuffle(activities)


            prompt = pro_desc + "\n\nEnsure the generated model uses the following activity labels (please also note upper and lower case): "+", ".join(activities)

            gen = LLMProcessModelGenerator(process_description=prompt, api_key=api_key, api_url=api_url,
                                           openai_model=model)

            conv = gen.get_conversation()
            F = open(target_conv, "w")
            json.dump(conv, F, indent=2)
            F.close()

            code = gen.get_code()
            F = open(target_code, "w")
            F.write(str(code))
            F.close()

            powl = gen.get_powl()
            net, im, fm = pm4py.convert_to_petri_net(powl)
            pm4py.write_pnml(net, im, fm, target_pn)
        except:
            traceback.print_exc()
            continue

#gen = LLMProcessModelGenerator(process_description="Purchase-to-Pay process", api_key=api_key, api_url=api_url, openai_model="Qwen/QwQ-32B-Preview")

#print(gen.get_conversation())
#print("\n\n")
#print(gen.get_powl())
#print("\n\n")
#print(gen.get_code())
#print("\n\n")

