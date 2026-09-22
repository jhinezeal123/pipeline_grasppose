"""Depth_Estimate — KHAI BAO giao dien, KHONG dinh nghia.

Xem Object_Detection.py de biet quy uoc chung cua 4 module.
Hien thuc cu the nam trong grasppose/adapters.py.
"""

from abc import ABC, abstractmethod


class Depth_Estimate(ABC):
    """Uoc luong do sau + dung point cloud tu MOT anh RGB.

    Vi du hien thuc: MoGe-3 (Ruicheng/moge-3-vitl).

    LUU Y QUAN TRONG:
      - Model chay tren TOAN ANH, khong chay tren mask. Mask chi duoc dung
        o buoc sau, de chon phan point cloud can cho GraspNess.
      - Khong nap them trong so dinov2 ben ngoai. MoGeModel.from_pretrained()
        da nap san encoder da fine-tune; nap chong len se GHI DE encoder va
        lam sai ty le do sau (da do duoc: 359% so voi 12%).
    """

    #: Duong dan trong so. Vi du: "model/moge-3-vitl"
    model_path = None

    #: Thiet bi chay. Vi du: "cuda", "cpu"
    device = None

    @abstractmethod
    def prepare(self, image, fov_x=None):
        """Nap trong so len VRAM va tien xu ly anh dau vao.

        Tham so:
            image : numpy uint8 (H, W, 3), kenh mau RGB
            fov_x : float hoac None.
                    - Anh THAT (khong biet truong nhin): de None. Model tu
                      uoc luong intrinsics.
                    - Anh RENDER tu MoJoCo (biet truoc fovy cua camera):
                      PHAI truyen vao, vi model neu tu doan se doan 60 do va
                      lam cloud bi keo gian theo chieu ngang.
                      Cong thuc: fov_x = 2*atan(tan(radians(fovy)/2) * W/H)

        Tra ve:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Chay suy dien, tra ve do sau + point cloud.

        Tra ve:
            dict voi cac khoa:
                "depth"      : numpy float32 (H, W), don vi MET. Pixel khong
                               hop le da duoc thay bang 0.0.
                "points"     : numpy float32 (H, W, 3), toa do 3D trong he
                               camera OpenCV (x phai, y xuong, z toi truoc),
                               don vi MET.
                "intrinsics" : numpy float32 (3, 3), da doi sang PIXEL
                               (khong con dang chuan hoa).
                "fov_x_deg"  : float, truong nhin ngang theo do.
                "reason"     : str hoac None.

        KHONG duoc raise. Neu that bai tra ve reason khac None.
        """
        raise NotImplementedError

    @abstractmethod
    def release(self):
        """Nha model khoi VRAM."""
        raise NotImplementedError
