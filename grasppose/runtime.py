"""Runtime/resource helpers. Model adapters own their state; this module owns CUDA cleanup."""

import gc

def _log(msg):
    print("[pipeline] %s" % msg, flush=True)


def _vram(tag=""):
    """In muc VRAM dang dung — de nhin ro phase nao con giu model."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        a = torch.cuda.memory_allocated() / 2**20
        r = torch.cuda.memory_reserved() / 2**20
        _log("  VRAM%s: da cap phat %.0f MB / da giu %.0f MB" % (tag, a, r))
    except Exception:
        pass


def _free(obj, *attrs):
    """Nha model khoi VRAM: xoa attribute -> gc -> empty_cache.

    Phai xoa attribute TRUOC khi gc/empty_cache, khong phai sau. Ban cu lam
    `del o` tren tung doi so — nhung do chi xoa ten cuc bo trong vong lap, con
    tuple tham so VA chinh `self.model` van giu reference. Nen gc.collect() va
    empty_cache() chay luc model CHUA duoc giai phong, tuc la khong thu hoi duoc
    gi; chi den khi ham return va dong `self.model = ... = None` chay sau do thi
    moi nha — nhung luc do da khong con empty_cache nua.

    Doi sang nhan (obj, ten_attr...): dat attribute ve None ngay tai day, roi moi
    gc + empty_cache. `obj` giu ten trong suot ham la khong sao — quan trong la
    attribute (tham chieu THAT toi model) da bi cat.
    """
    for name in attrs:
        setattr(obj, name, None)
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        # synchronize truoc: empty_cache() can moi kernel dung model da chay xong,
        # neu khong thi bo nho co the chua kip duoc tra ve allocator.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()



def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


