import copy

import numpy as np


class QuantumGene:
    # 2 gene แทน 1 gate (alpha|0> + beta|1>)
    # และทำหน้าที่หมุนมุม update ตามตาราง
    def __init__(self):
        # เริ่มที่มุม pi/4 (Uniform Superposition)
        self.theta = np.pi / 4
        self.update_amplitudes()

    def update_amplitudes(self):
        # คำนวณมุม alpha, beta ใหม่จาก theta
        self.alpha = np.cos(self.theta)
        self.beta = np.sin(self.theta)

    def observe(self):
        # วัดผลลัพธ์จากโครโมโซม alpha|0> + beta|1>
        prob_1 = self.beta**2
        return 1 if np.random.rand() < prob_1 else 0

    def rotate(self, current_bit, best_bit, fitness_current, fitness_best):
        # F>=Fbest ?
        is_current_F_better = fitness_current >= fitness_best

        d_theta = 0.0
        sign = 0
        delta = 0.03 * np.pi
        eps = 1e-6  # ดักกรณีใกล้ศูนย์มากๆ

        # ตารางหมุนมุม
        if current_bit == 0 and best_bit == 0:
            d_theta = 0.0
        elif current_bit == 0 and best_bit == 1:
            if not is_current_F_better:
                d_theta = 0.0
            else:
                d_theta = delta
                # sigh
                if abs(self.beta) < eps:
                    sign = 0
                elif abs(self.alpha) < eps:
                    sign = np.random.choice([-1, 1])  # sign +-1
                elif self.alpha * self.beta > 0:
                    sign = -1
                elif self.alpha * self.beta < 0:
                    sign = 1
        elif current_bit == 1 and best_bit == 0:
            if not is_current_F_better:
                d_theta = delta
                # sign
                if abs(self.beta) < eps:
                    sign = 0
                elif abs(self.alpha) < eps:
                    sign = np.random.choice([-1, 1])  # sign +-1
                elif self.alpha * self.beta > 0:
                    sign = -1
                elif self.alpha * self.beta < 0:
                    sign = 1
            else:
                d_theta = delta
                # sign
                if abs(self.alpha) < eps:
                    sign = 0
                elif abs(self.beta) < eps:
                    sign = np.random.choice([-1, 1])  # sign +-1
                elif self.alpha * self.beta > 0:
                    sign = 1
                elif self.alpha * self.beta < 0:
                    sign = -1
        elif current_bit == 1 and best_bit == 1:
            d_theta = delta
            # sign
            if abs(self.alpha) < eps:
                sign = 0
            elif abs(self.beta) < eps:
                sign = np.random.choice([-1, 1])  # sign +-1
            elif self.alpha * self.beta > 0:
                sign = 1
            elif self.alpha * self.beta < 0:
                sign = -1

        # ปรับมุม theta
        self.theta += sign * d_theta
        # ทำให้อยู่ในช่วง 0 - 2pi เพื่อไม่ให้ยิ่งหมุนยิ่งเกิน
        self.theta = self.theta % (2 * np.pi)
        # ***************************************
        self.update_amplitudes()


class QuantumChromosome:
    # 2gene แทน 1 gate ，num_gates = จำนวน gate
    # genes = หลาย gene = คือโครโมโซมที่แทน QCNN 1 วงจร = 1 คน
    def __init__(self, num_gates):
        self.num_gates = num_gates

        # สร้าง gene จำนวน num_gates*2 เพื่อ 2 gene = 1 gate
        self.genes = [QuantumGene() for _ in range(num_gates * 2)]

        self.binary_code = []  # เก็บ [0, 1, 0, 0, ...] ความยาว num_gates * 2
        self.structure_code = []  # เก็บ [2, 0, ...]       ความยาว num_gates (ส่งให้ Builder)
        self.fitness = 0.0  # ความแม่นของวงจร

    def collapse(self):
        # วัดโครโมโซมทั้งหมดในประชากร แล้วแปลงเป็นเกต
        # 1. วัด gene แต่ละตัว binary
        self.binary_code = [gene.observe() for gene in self.genes]
        # 2. แปลง Binary คู่ เป็น รหัสเกต (0-3)

        self.structure_code = [
            (self.binary_code[i] << 1) | self.binary_code[i + 1]
            for i in range(0, len(self.binary_code), 2)
        ]
        return self.structure_code

    def update_genes(self, global_best_chromosome):
        # หมุนมุมโดยเทียบ คน(วงจร)นี้ กับ คน(วงจร)ที่ Best
        best_binary = global_best_chromosome.binary_code
        best_fitness = global_best_chromosome.fitness

        # เทียบทีละบิต
        for i, gene in enumerate(self.genes):
            gene.rotate(
                current_bit=self.binary_code[i],
                best_bit=best_binary[i],
                fitness_current=self.fitness,
                fitness_best=best_fitness,
            )

    def copy(self):
        """ฟังก์ชันสำหรับ copy ตัวเอง (ใช้ตอนเก็บ Global Best)"""
        new_instance = QuantumChromosome(self.num_gates)
        new_instance.genes = copy.deepcopy(self.genes)
        new_instance.binary_code = list(self.binary_code)
        new_instance.structure_code = list(self.structure_code)
        new_instance.fitness = self.fitness
        return new_instance
