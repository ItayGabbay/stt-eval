from utils.data_helper import load_blobs_from_cotainer
from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataloader_path', default=f'dataset/reupload/test')
    parser.add_argument('--local_path', default=f'../../stt_project/dataset')
    parser.add_argument('--file_type', default='.wav')

    args = parser.parse_args()

    dataloader = load_blobs_from_cotainer(args.dataloader_path)
    
    if args.file_type == '.wav':
        for i, (name, data) in enumerate(dataloader):
            local_file_path = os.path.join(args.local_path, name)
            with open(local_file_path, "wb") as local_file:
                data.readinto(local_file)
    
    elif args.file_type == '.txt':
        for i, (name, data) in enumerate(dataloader):
            local_file_path = os.path.join(args.local_path, name)
            with open(local_file_path, "w", encoding="utf-8-sig") as local_file:
                text = data.readall().decode("utf-8-sig")
                local_file.write(text)



    