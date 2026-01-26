import numpy as np
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.utils import algorithm_globals


class DataManager:
    def __init__(self, num_images, test_size=0.3, random_seed=12345):
        self.num_images = num_images
        self.test_size = test_size
        algorithm_globals.random_seed = random_seed
        self.train_images = None
        self.test_images = None
        self.train_labels = None
        self.test_labels = None

    def generate_synthetic_dataset(self):
        images = []
        labels = []
        hor_array = np.zeros((6, 8))
        ver_array = np.zeros((4, 8))

        # สร้าง pattern แนวนอน
        j = 0
        for i in range(0, 7):
            if i != 3:  # มีแค่2*4พิกเซล ดังนั้น i=3 จะเป็นตัวต่อแถวบนกับแถวล่าง
                hor_array[j][i] = np.pi / 2
                hor_array[j][i + 1] = np.pi / 2
                j += 1

        # สร้าง pattern แนวตั้ง
        j = 0
        for i in range(0, 4):
            ver_array[j][i] = np.pi / 2
            ver_array[j][i + 4] = np.pi / 2
            j += 1

        # สุ่มสร้างรูป
        for _ in range(self.num_images):
            rng = algorithm_globals.random.integers(0, 2)
            if rng == 0:  # แนวนอน
                labels.append(-1)
                random_image = algorithm_globals.random.integers(0, 6)
                base_image = np.array(hor_array[random_image])
            elif rng == 1:  # แนวตั้ง
                labels.append(1)
                random_image = algorithm_globals.random.integers(0, 4)
                base_image = np.array(ver_array[random_image])

            # ใส่ noise
            noise = algorithm_globals.random.uniform(0, np.pi / 4, size=8)  # สุ่มค่าจาก0-pi/4ใส่
            final_image = np.where(base_image == 0, noise, base_image)
            images.append(final_image)

        self.images = np.array(images)
        self.labels = np.array(labels)  # Fixed typo: changed self.label to self.labels

        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(
            self.images, self.labels, test_size=self.test_size, random_state=246
        )
        print(
            f"Dataset Generated: {len(self.train_images)} Train samples, {len(self.test_images)} Test samples"
        )
