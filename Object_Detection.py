"""Object_Detection — KHAI BAO giao dien, KHONG dinh nghia.

File nay chi mo ta mot class de sau nay thao/lap model khac vao.
Hien thuc cu the nam trong pipeline.py (lop con ke thua class nay).

Quy uoc chung cho ca 4 module:
  - model_path : thuoc tinh, duong dan trong so tren dia
  - prepare()  : nap trong so len VRAM + tien xu ly dau vao
  - inference(): chay suy dien, tra ket qua
  - release()  : nha VRAM (bat buoc, vi pipeline chay theo 3 phase)
"""

from abc import ABC, abstractmethod


class Object_Detection(ABC):
    """Phat hien vat the tu anh + cau lenh van ban.

    Vi du hien thuc: Grounding-DINO.
    """

    #: Duong dan trong so. Vi du: "model/grounding-dino-tiny"
    model_path = None

    #: Thiet bi chay. Vi du: "cuda", "cpu"
    device = None

    @abstractmethod
    def prepare(self, image, prompt):
        """Nap trong so len VRAM va tien xu ly anh dau vao.

        Tham so:
            image  : numpy uint8 (H, W, 3), kenh mau RGB
            prompt : str, cau lenh van ban, vi du "a little bag"

        Tra ve:
            self (de goi noi tiep .inference())

        Sau khi goi, trong so PHAI nam tren VRAM san sang cho inference().
        """
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Chay suy dien, tra ve hop bao vat the.

        Tra ve:
            dict voi cac khoa:
                "boxes"  : numpy float32 (N, 4), toa do PIXEL theo thu tu
                           [x_min, y_min, x_max, y_max]
                "scores" : numpy float32 (N,)
                "labels" : list[str] (N,), nhan van ban cua tung hop
                "reason" : str hoac None. Neu None tuc la thanh cong; neu khac
                           None tuc la KHONG tim duoc hop nao va day la ly do
                           (vi du "no box for prompt").

        KHONG duoc raise khi khong tim thay vat. Tra ve boxes rong + reason.
        """
        raise NotImplementedError

    @abstractmethod
    def release(self):
        """Nha model khoi VRAM (garbage collect + torch.cuda.empty_cache()).

        Sau khi goi, khong duoc goi lai inference() truoc khi prepare() lai.
        """
        raise NotImplementedError
