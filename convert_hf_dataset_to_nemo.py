import json
import os
from datasets import load_dataset, Audio
import soundfile
import librosa
from tqdm import tqdm
from multiprocessing import Pool
import uuid
import glob

def create_nemo_manifest(dataset_name, subset_name, split_name, text_column_name, batch_size = 1000, output_dir="./dataset"):
    token_hf = ""
    if subset_name:
        dataset = load_dataset(dataset_name, subset_name, split=split_name, streaming=True, token=token_hf, trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name, subset_name, split=split_name, streaming=True, token=token_hf, trust_remote_code=True)

    columns_to_remove = ['audio', text_column_name]
    for col in columns_to_remove:
        if col not in dataset.features:
            dataset = dataset.remove_columns([col])

    if 'text' not in dataset.features:
        dataset = dataset.rename_column(text_column_name, 'text')

    dataset = dataset.cast_column("audio", Audio(16000, mono=True))

    manifest_dataset_name = dataset_name.replace("/", "-")
    manifest_path = os.path.join(output_dir, f"{split_name}_manifest_{manifest_dataset_name}.json")
    # Tạo thư mục output
    output_dir = os.path.join(output_dir, split_name)
    os.makedirs(output_dir, exist_ok=True)

    dataset_index = 0
    manifest_entries = []
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        for sample in tqdm(dataset, desc=f"Processing {split_name}"):
            try:
                uuid4 = uuid.uuid4()
                audio_filename = f"{split_name}_{uuid4}.wav"
                audio_filepath = os.path.join(output_dir, audio_filename)

                soundfile.write(
                    audio_filepath,
                    sample['audio']['array'],
                    samplerate=16000,
                    format='wav'
                )

                duration = librosa.get_duration(
                    y=sample['audio']['array'],
                    sr=sample['audio']['sampling_rate']
                )

                manifest_entry = {
                    "audio_filepath": os.path.abspath(audio_filepath),
                    "text": sample['text'],
                    "duration": duration
                }
                manifest_entries.append(manifest_entry)
                dataset_index += 1

                if len(manifest_entries) >= batch_size:
                    manifest_file.writelines(
                        [json.dumps(entry, ensure_ascii=False) + '\n' for entry in manifest_entries]
                    )
                    manifest_entries = []

            except Exception as e:
                print(f"Error processing sample {dataset_index}: {e}")
                dataset_index += 1
                continue

        if manifest_entries:
            manifest_file.writelines(
                [json.dumps(entry, ensure_ascii=False) + '\n' for entry in manifest_entries]
            )


    print(f"Đã xử lý {dataset_index} samples và lưu manifest tại: {manifest_path}")


def create_nemo_manifest_wrapper(task_params):
    """Wrapper function để unpack arguments cho multiprocessing"""
    return create_nemo_manifest(**task_params)


def merge_manifest():
    folder = './dataset'

    all_files = glob.glob(os.path.join(folder, '*_manifest_*.json'))

    base_names = set()
    for file in all_files:
        basename = os.path.basename(file)
        base = basename.split('_manifest_')[0] + '_manifest'
        base_names.add(base)

    for base in base_names:
        part_files = sorted(glob.glob(os.path.join(folder, f'{base}_*.json')))
        with open(os.path.join(folder, f'{base}.json'), 'w', encoding='utf-8') as outfile:
            for fname in part_files:
                with open(fname, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
        print(f'Đã merge {len(part_files)} file thành {base}.json')


def run_parallel_processing():
    """Chạy song song 3 task create_nemo_manifest"""

    # Định nghĩa các task parameters
    tasks = [
        {
            'dataset_name': 'mozilla-foundation/common_voice_17_0',
            'subset_name': 'vi',
            'split_name': 'train',
            'text_column_name': 'sentence',
        },
        {
            'dataset_name': 'mozilla-foundation/common_voice_17_0',
            'subset_name': 'vi',
            'split_name': 'test',
            'text_column_name': 'sentence',
        },
        {
            'dataset_name': 'mozilla-foundation/common_voice_17_0',
            'subset_name': 'vi',
            'split_name': 'validation',
            'text_column_name': 'sentence',
        },

        {
            'dataset_name': 'AILAB-VNUHCM/vivos',
            'subset_name': None,
            'split_name': 'train',
            'text_column_name': 'sentence',
        },
        {
            'dataset_name': 'AILAB-VNUHCM/vivos',
            'subset_name': None,
            'split_name': 'test',
            'text_column_name': 'sentence',
        },
        {
            'dataset_name': 'capleaf/viVoice',
            'subset_name': None,
            'split_name': 'train',
            'text_column_name': 'text',
        },
        {
            'dataset_name': 'NhutP/VietSpeech',
            'subset_name': None,
            'split_name': 'train',
            'text_column_name': 'transcription',
        },
        {
            'dataset_name': 'doof-ferb/fpt_fosd',
            'subset_name': None,
            'split_name': 'train',
            'text_column_name': 'transcription',
        },
        {
            'dataset_name': 'doof-ferb/infore1_25hours',
            'subset_name': None,
            'split_name': 'train',
            'text_column_name': 'transcription',
        },
        {
            'dataset_name': 'google/fleurs',
            'subset_name': 'vi_vn',
            'split_name': 'train',
            'text_column_name': 'transcription',
        },
        {
            'dataset_name': 'google/fleurs',
            'subset_name': 'vi_vn',
            'split_name': 'validation',
            'text_column_name': 'transcription',
        },
        {
            'dataset_name': 'google/fleurs',
            'subset_name': 'vi_vn',
            'split_name': 'test',
            'text_column_name': 'transcription',
        },
        {
            'dataset_name': 'linhtran92/viet_youtube_asr_corpus_v2',
            'subset_name': None,
            'split_name': 'train',
            'text_column_name': 'w2v2_transcription',
        },
        {
            'dataset_name': 'linhtran92/viet_youtube_asr_corpus_v2',
            'subset_name': None,
            'split_name': 'test',
            'text_column_name': 'w2v2_transcription',
        },
        {
            'dataset_name': 'linhtran92/viet_bud500',
            'subset_name': None,
            'split_name': 'train',
            'text_column_name': 'transcription',
        },
        {
            'dataset_name': 'linhtran92/viet_bud500',
            'subset_name': None,
            'split_name': 'validation',
            'text_column_name': 'transcription',
        },
        {
            'dataset_name': 'linhtran92/viet_bud500',
            'subset_name': None,
            'split_name': 'test',
            'text_column_name': 'transcription',
        },
        {
            'dataset_name': 'thivux/phoaudiobook',
            'subset_name': None,
            'split_name': 'train',
            'text_column_name': 'text',
        },
        {
            'dataset_name': 'thivux/phoaudiobook',
            'subset_name': None,
            'split_name': 'validation',
            'text_column_name': 'text',
        },
        {
            'dataset_name': 'thivux/phoaudiobook',
            'subset_name': None,
            'split_name': 'test',
            'text_column_name': 'text',
        },
    ]

    # Chạy song song với multiprocessing
    print(f"Bắt đầu xử lý song song {len(tasks)} splits...")
    with Pool(processes=len(tasks)) as pool:
        pool.map(create_nemo_manifest_wrapper, tasks)

    print("Hoàn thành xử lý tất cả splits!")
    merge_manifest()


if __name__ == "__main__":
    run_parallel_processing()
