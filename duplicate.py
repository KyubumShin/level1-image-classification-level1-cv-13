import pandas as pd 
import os
from torchvision.io import read_image
from torchvision.io import write_jpeg

def age_group(x):
    if x < 30:
        return "0<age<30"
    elif x < 59:
        return "30<=age<60"
    else:
        return "60<=age"


def data_pipe(df: pd.DataFrame) -> pd.DataFrame:
    df["age_group"] = df["age"].apply(age_group)
    return df

image_dir = "../../input/data/train/images"
train_df = pd.read_csv("../../input/data/train/train.csv")
train_df = train_df.drop('race', axis = 1)

train_df = data_pipe(train_df)

#id gender agegroup age path 
#temp = pd.DataFrame(columns=['id', 'gender', 'age', 'age_group', 'path'])
_file_names = [ "mask1",  "mask2",  "mask3",  "mask4",  "mask5", "incorrect_mask", "normal"]


temp = 0 
for line in train_df.iloc:
    for file in list(os.listdir(os.path.join(image_dir, line['path']))):
        _file_name, ext = os.path.splitext(file)
        if not _file_name  in _file_names:
            continue
            
        
        if line['age'] > 58: # 59세부터. deta 개수의 X 5 해줘야 데이터 균형 

            #파일 이름 mask.jpg --> mask-0.jpg , mask-1.jpg, mask-2.jpg.. 형식으로 생성 저장 
            #
            img = read_image(os.path.join(image_dir,line["path"],file))    
            for i in range(4): # 4 의 숫자에 따라 데이터 복사 횟수 
                save_path = os.path.join(image_dir,line['path'] ,_file_name+'-'+str(i)+ext)
                if not os.path.exists(save_path):
                    write_jpeg(img, filename = save_path, quality=100)
        

'''
파일 삭제
for i in range(4):
    remove_path = os.path.join(image_dir,line['path'] ,_file_name+'_'+str(3)+ext)
        if os.path.exists(remove_path):
            os.remove(remove_path)
        else:
            pass
'''