"""
IARA
"""
import typing
import numpy as np

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_signal
import lps_ml.databases.loader as lps_loader
import lps_ml.databases.iara as lps_iara
import lps_ml.processors as lps_proc


def main():
    """Main function for the dataset info tables."""

    df = lps_iara.Collection.A.to_df()[:3]
    file_ids = df["ID"].tolist()

    base_dir = "/data/IARA"
    file_loader = lps_loader.AudioFileLoader.iara(base_dir)

    processor = lps_proc.WindowingResampler(
        fs = lps_qty.Frequency.khz(16),
        duration = lps_qty.Time.s(20),
        overlap = lps_qty.Time.s(10)
    )

    # h5_path = "./results"
    # dataloader = lps_loader.AudioDataLoader(
    #     file_dir=h5_path,
    #     file_loader=file_loader,
    #     file_processor=processor,
    #     file_ids=file_ids
    # )

    # print(f"\nTotal windows processed: {len(dataloader)}")

    # for fid in file_ids:
    #     proc_ids = dataloader.map_file_ids_to_processed_ids([fid])
    #     print(f"{fid} -> {proc_ids}")

    # first_fid = file_ids[0]
    # first_proc_ids = dataloader.map_file_ids_to_processed_ids([first_fid])
    # if first_proc_ids:
    #     batch = dataloader.load(first_proc_ids[:2])
    #     print(f"\nBatch shape {batch.shape} (torch.Tensor)")

if __name__ == "__main__":
    main()
