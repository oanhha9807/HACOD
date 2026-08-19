from pathlib import Path

import yaml

import shutil
from pathlib import Path

from pathlib import Path
from collections import defaultdict


from ultralytics import YOLO


class YOLOEmbeddingExtractor:

    def __init__(
        self,
        model_path,
        layer_idx=9,
        device="cuda"
    ):

        self.device = device

        self.model = YOLO(model_path).model
        self.model.to(device)
        self.model.eval()

        self.feature = None

        self.hook = self.model.model[layer_idx].register_forward_hook(
            self._hook_fn
        )

    def _hook_fn(self, module, inp, out):

        if isinstance(out, (tuple, list)):
            out = out[0]

        self.feature = out

    def preprocess(
        self,
        img_path,
        imgsz=640
    ):

        img = cv2.imread(str(img_path))

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (imgsz, imgsz)
        )

        img = img.astype(np.float32) / 255.0

        img = torch.from_numpy(img)

        img = img.permute(2, 0, 1)

        img = img.unsqueeze(0)

        return img.to(self.device)

    @torch.no_grad()
    def extract(
        self,
        img_path,
        imgsz=640
    ):

        x = self.preprocess(
            img_path,
            imgsz
        )

        _ = self.model(x)

        feat = self.feature

        feat = F.adaptive_avg_pool2d(
            feat,
            (1, 1)
        )

        feat = feat.flatten(1)

        feat = F.normalize(
            feat,
            dim=1
        )

        return feat.squeeze(0).cpu().numpy()

    def close(self):
        self.hook.remove()




def compute_class_prototypes_from_memory(
    memory_root,
    feature_model_path,
    layer_idx=9,
    imgsz=640
):

    memory_root = Path(memory_root)

    img_dir = memory_root / "images/train"
    lab_dir = memory_root / "labels/train"

    if not img_dir.exists() or not lab_dir.exists():
        return {}

    class_to_images = collect_images_by_class(
        img_dir,
        lab_dir
    )

    extractor = YOLOEmbeddingExtractor(
        feature_model_path,
        layer_idx=layer_idx,
        device=device
    )

    prototypes = {}

    for cls_id, images in class_to_images.items():

        embeddings = []

        for img_path in images:

            try:

                feat = extractor.extract(
                    img_path,
                    imgsz
                )

                if np.isnan(feat).any():
                    continue

                embeddings.append(
                    feat
                )

            except Exception:
                continue

        if len(embeddings) == 0:
            continue

        embeddings = np.stack(
            embeddings
        )

        prototype = np.mean(
            embeddings,
            axis=0
        )

        prototype = (
            prototype /
            (
                np.linalg.norm(
                    prototype
                ) + 1e-8
            )
        )

        prototypes[cls_id] = prototype

    extractor.close()

    return prototypes



def embedding_uncertainty(
    feat,
    prototype
):

    sim = np.dot(
        feat,
        prototype
    )

    return 1.0 - sim



def select_sota_replay_exemplars_for_class(
    image_paths,
    class_id,
    memory_per_class,
    feature_model_path,
    old_prototype=None,
    alpha=0.45,
    beta=0.25,
    gamma=0.20,
    delta=0.10,
    candidate_size=50,
    imgsz=640,
    layer_idx=9
):

    if len(image_paths) <= memory_per_class:
        return list(image_paths)

    extractor = YOLOEmbeddingExtractor(
        feature_model_path,
        layer_idx=layer_idx,
        device=device
    )
    print('*****************Using YOLOEmbeddingExtractor **************')

    features = []
    valid_images = []

    # =====================================
    # EMBEDDING EXTRACTION
    # =====================================
    for img_path in image_paths:

        try:

            feat = extractor.extract(
                img_path,
                imgsz
            )

            if np.isnan(feat).any():
                continue

            features.append(feat)

            valid_images.append(img_path)

        except Exception:
            continue

    extractor.close()

    if len(valid_images) <= memory_per_class:
        return valid_images

    features = np.stack(features)

    features = (
        features /
        (
            np.linalg.norm(
                features,
                axis=1,
                keepdims=True
            ) + 1e-8
        )
    )

    # =====================================
    # CURRENT PROTOTYPE
    # =====================================
    current_proto = np.mean(
        features,
        axis=0
    )

    current_proto = (
        current_proto /
        (
            np.linalg.norm(
                current_proto
            ) + 1e-8
        )
    )

    # =====================================
    # OLD PROTOTYPE
    # =====================================
    if old_prototype is not None:

        old_proto = (
            old_prototype /
            (
                np.linalg.norm(
                    old_prototype
                ) + 1e-8
            )
        )

    else:
        old_proto = None

    # =====================================
    # UNCERTAINTY
    # distance from prototype
    # =====================================
    uncertainties = []

    for feat in features:

        unc = np.linalg.norm(
            feat - current_proto
        )

        uncertainties.append(
            unc
        )

    uncertainties = np.array(
        uncertainties
    )

    uncertainties = (
        uncertainties /
        (
            uncertainties.max()
            + 1e-8
        )
    )

    # =====================================
    # INITIAL EXEMPLAR
    # =====================================
    scores = features @ current_proto

    init_idx = int(
        np.argmax(scores)
    )

    chosen = [init_idx]

    remaining = list(
        range(len(features))
    )

    remaining.remove(
        init_idx
    )

    # =====================================
    # GREEDY SELECTION
    # =====================================
    while len(chosen) < memory_per_class:

        best_idx = None

        best_score = -1e9

        if len(remaining) > candidate_size:

            rng = random.Random(42)

            candidates = rng.sample(
                remaining,
                candidate_size
            )

        else:

            candidates = remaining

        for idx in candidates:

            feat = features[idx]

            # -------------------------
            # representative
            # -------------------------
            rep = np.dot(
                feat,
                current_proto
            )

            # -------------------------
            # stability
            # -------------------------
            if old_proto is not None:

                align = np.dot(
                    feat,
                    old_proto
                )

            else:

                align = 0.0

            # -------------------------
            # diversity
            # -------------------------
            div = min(
                np.linalg.norm(
                    feat - features[j]
                )
                for j in chosen
            )

            # -------------------------
            # uncertainty
            # -------------------------
            unc = uncertainties[idx]

            # -------------------------
            # final score
            # -------------------------
            score = (
                alpha * rep
                + beta * align
                + gamma * div
                + delta * unc
            )
            # score = (
            #         0.5 * rep +
            #         0.2 * align +
            #         0.2 * div +
            #         0.2 * unc
            # )#best resnet




            if score > best_score:

                best_score = score

                best_idx = idx

        if best_idx is None:
            break

        chosen.append(
            best_idx
        )

        remaining.remove(
            best_idx
        )

    return [
        valid_images[i]
        for i in chosen
    ]