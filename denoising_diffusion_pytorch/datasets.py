from skimage import io
from torch.utils.data import Dataset
from pathlib import Path, PurePath
import pandas as pd
import numpy as np
import os


class SingleCell(Dataset):
    def __init__(
        self,
        image_path,
        dataframe_path,
        transforms=None,
        image_size=400,
        cell_component="both",
    ):
        self.image_path = image_path
        self.dataframe_path = dataframe_path
        self.p = Path(self.image_path)
        self.df = pd.read_csv(dataframe_path)
        self.transforms = transforms
        self.image_size = image_size
        self.cell_component = cell_component
        choices = ["cell", "nuc", "both"]
        assert cell_component in choices, f"Please choose one of {choices}."

        self.new_df = self.df[
            (self.df.xDim <= self.image_size)
            & (self.df.yDim <= self.image_size)
            & (self.df.zDim <= self.image_size)
        ].reset_index(drop=True)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        serial_number = self.new_df.loc[idx, "serialNumber"]
        if self.cell_component == "cell":
            component_path = "stacked_intensity_cell"
            path = os.path.join(
                self.image_path, plate_num, component_path, serial_number + ".tif"
            )
            image = io.imread(path)
        elif self.cell_component == "nuc":
            component_path = "stacked_intensity_nucleus"
            path = os.path.join(
                self.image_path, plate_num, component_path, serial_number + ".tif"
            )
            image = io.imread(path)
        else:
            cell_component_path = "stacked_intensity_cell"
            try:
                cell_path = os.path.join(
                    self.image_path,
                    plate_num,
                    cell_component_path,
                    serial_number + ".tif",
                )
                cell_image = io.imread(cell_path)
            except:
                serial_number = serial_number.swapcase()
                cell_path = os.path.join(
                    self.image_path,
                    plate_num,
                    cell_component_path,
                    serial_number + ".tif",
                )
                cell_image = io.imread(cell_path)

            nuc_component_path = "stacked_intensity_nucleus"
            nuc_path = os.path.join(
                self.image_path, plate_num, nuc_component_path, serial_number + ".tif"
            )
            nuc_image = io.imread(nuc_path)
            image = np.stack((cell_image, nuc_image))

        if self.transforms:
            image = self.transforms(image)

        return image, treatment, serial_number, plate_num


def pad_img(img, new_size):
    new_z, new_y, new_x = new_size, new_size, new_size
    z = img.shape[0]
    y = img.shape[1]
    x = img.shape[2]
    delta_z = new_z - z
    delta_y = new_y - y
    delta_x = new_x - x

    if delta_z % 2 == 1:
        z_padding = (delta_z // 2, delta_z // 2 + 1)
    else:
        z_padding = (delta_z // 2, delta_z // 2)

    if delta_y % 2 == 1:
        y_padding = (delta_y // 2, delta_y // 2 + 1)
    else:
        y_padding = (delta_y // 2, delta_y // 2)

    if delta_x % 2 == 1:
        x_padding = (delta_x // 2, delta_x // 2 + 1)
    else:
        x_padding = (delta_x // 2, delta_x // 2)

    padded_data = np.pad(img, (z_padding, y_padding, x_padding), "constant")
    return padded_data
