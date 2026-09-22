"""GraspNess — KHAI BAO giao dien, KHONG dinh nghia.

Xem Object_Detection.py de biet quy uoc chung cua 4 module.
Hien thuc cu the nam trong grasppose/adapters.py.
"""

from abc import ABC, abstractmethod


class GraspNess(ABC):
    """Sinh cac tu the gap (grasp pose) tu point cloud cua vat the.

    Vi du hien thuc: GraspNet + Graspness (checkpoint graspness_realsense.pth).

    LUU Y QUAN TRONG:
      - Dau vao la point cloud lay tu MASK THUAN, khong phai hinh bao (bbox)
        mo rong. Ban dung bbox +-2cm tung de lot diem cua vat the khac trong
        canh vao cloud va sinh ra grasp nam ngoai vat.
    """

    #: Duong dan trong so. Vi du: "model/graspness_reckpt.pth"
    model_path = None

    #: Thiet bi chay. Vi du: "cuda", "cpu"
    device = None

    @abstractmethod
    def prepare(self, points, num_point=15000):
        """Nap trong so len VRAM va tien xu ly point cloud.

        Tham so:
            points    : numpy float32 (M, 3), toa do 3D trong he camera
                        OpenCV, don vi MET
            num_point : int, so diem lay mau de dua vao model

        Tra ve:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Chay suy dien, tra ve tap tu the gap.

        Tra ve:
            dict voi cac khoa:
                "graspgroup" : numpy float64 (N, 17). Y nghia tung cot:
                                 [0]     score    - diem tin cay
                                 [1]     width    - khe mo ngon kep (MET)
                                 [2]     height   - chieu cao ngon (MET)
                                 [3]     depth    - do sau ngam (MET)
                                 [4:13]  R        - ma tran xoay 3x3 (flatten,
                                                    theo thu tu row-major)
                                 [13:16] translation - tam kep (MET)
                                 [16]    obj_id
                               He quy chieu cua kep: truc +x la huong tiep can.
                "reason"     : str hoac None.

        KHONG duoc raise.
        """
        raise NotImplementedError

    @abstractmethod
    def release(self):
        """Nha model khoi VRAM."""
        raise NotImplementedError
