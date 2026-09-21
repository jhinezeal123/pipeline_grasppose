"""Segmentation — KHAI BAO giao dien, KHONG dinh nghia.

Xem Object_Detection.py de biet quy uoc chung cua 4 module.
Hien thuc cu the nam trong pipeline.py.
"""

from abc import ABC, abstractmethod


class Segmentation(ABC):
    """Tao mat na (mask) pixel cua vat the, dua tren hop cua module truoc.

    Vi du hien thuc: SAM (Segment Anything).
    """

    #: Duong dan trong so. Vi du: "model/sam-vit-base"
    model_path = None

    #: Thiet bi chay. Vi du: "cuda", "cpu"
    device = None

    @abstractmethod
    def prepare(self, image, boxes):
        """Nap trong so len VRAM va tien xu ly anh + hop dau vao.

        Tham so:
            image : numpy uint8 (H, W, 3), kenh mau RGB
            boxes : numpy float32 (N, 4), toa do PIXEL [x_min, y_min, x_max, y_max]
                    thuong la ket qua cua Object_Detection.inference()

        Tra ve:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Chay suy dien, tra ve mat na cua vat the.

        Tra ve:
            dict voi cac khoa:
                "mask"    : numpy bool (H, W). True = pixel thuoc vat the.
                "iou"     : numpy float32 (N,), diem IoU cua tung ung vien
                "best"    : int, chi so ung vien duoc chon
                "n_pred"  : int, so ung vien model dua ra
                "reason"  : str hoac None, ly do neu that bai

        Neu khong co hop nao dau vao thi tra ve mask rong (toan False) va
        reason khac None. KHONG duoc raise.
        """
        raise NotImplementedError

    @abstractmethod
    def release(self):
        """Nha model khoi VRAM."""
        raise NotImplementedError
