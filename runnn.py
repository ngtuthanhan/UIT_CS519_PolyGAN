import os
file_train = os.listdir('./data/train/image')
file_test = os.listdir('./data/test/image')
file_res = os.listdir('./results/test/Stage3temp_res')
print(len(file_train))
print(len(file_test))
print(len(file_res))
file_train_not = []
file_test_not = []
file_train = [file[:6] for file in file_train]
file_test = [file[:6] for file in file_test]
file_res = [file[2:8] for file in file_res]

# for i in file_train:
#     if i not in file_res:
#         file_train_not.append(i)
#         os.system(f"python test.py --stage Stage3 --datamode train --model_image {i}_0.jpg --reference_image {i}_1.jpg ")

# for i in file_test:
#     if i not in file_res:
#         file_test_not.append(i)
#         os.system(f"python test.py --stage Stage3 --datamode test --model_image {i}_0.jpg --reference_image {i}_1.jpg ")

with open('./data/test_pairs_pc.txt', 'r') as f:
    txt = f.read()

lines = txt.split('\n')
for line in lines:
    if line == '':
        continue
    image_path = line.split()
    model_image = image_path[0]
    reference_image = image_path[1]
    os.system(f"python test.py --stage Stage3 --datamode test --model_image {model_image} --reference_image {reference_image} --results_Stage3 results/test/Stage3/true")
    

# print(len(file_train_not))
# print(len(file_test_not))

