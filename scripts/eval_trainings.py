"""Quick eval of all training run checkpoints on the Cassava val set."""
import torch, os, sys
sys.path.insert(0, 'src')

from torch.utils.data import DataLoader
from torchvision import transforms as T
from elysia_cv.data import CassavaDataset
from elysia_cv.models.classification.cassava.runtime import build_model
from elysia_cv.config import load_cv_config_with_object

val_ds = CassavaDataset(
    data_root='/data/hdd3/chuqi/data/cassava', split='val',
    transform=T.Compose([
        T.Resize(448), T.CenterCrop(448), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
val_ldr = DataLoader(val_ds, batch_size=64, num_workers=4, pin_memory=True)


@torch.no_grad()
def eval_ckpt(ckpt_path, model):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.cuda().eval()
    correct = total = 0
    for batch in val_ldr:
        imgs = batch['inputs'].cuda()
        labels = batch['gt_label'].cuda()
        out = model(imgs)
        logits = out['logits'] if isinstance(out, dict) else out
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.cpu()
    return correct / total


def build_from_exp(cfg_path):
    cfg = load_cv_config_with_object(cfg_path)
    d = {'model': {'type': cfg.model.type, 'backbone': dict(cfg.model.backbone)}}
    neck = getattr(cfg.model, 'neck', None)
    if neck is not None:
        d['model']['neck'] = dict(neck)
    head = getattr(cfg.model, 'head', None)
    if head is not None:
        d['model']['head'] = dict(head)
    return build_model(d)


def find_best_ckpt(wd):
    """Find best checkpoint: best_Accuracy > best > latest step."""
    for prefix in ['checkpoint_best_Accuracy', 'checkpoint_best']:
        path = f'{wd}/{prefix}.pt'
        if os.path.exists(path):
            return prefix, path

    step_files = sorted(
        [f for f in os.listdir(wd) if f.startswith('checkpoint_step_') and f.endswith('.pt')],
        key=lambda x: int(x.replace('checkpoint_step_', '').replace('.pt', '')),
    )
    if step_files:
        return step_files[-1].replace('.pt', ''), f'{wd}/{step_files[-1]}'
    return None, None


def get_max_step(wd):
    step_files = [f for f in os.listdir(wd)
                  if f.startswith('checkpoint_step_') and f.endswith('.pt')]
    if step_files:
        return max(int(f.replace('checkpoint_step_', '').replace('.pt', '')) for f in step_files)
    return 0


pairs = [
    ('cassava_simple_cnn',  'configs/experiments/cassava_simple_cnn.yaml'),
    ('cassava_resnet18',    'configs/experiments/cassava_resnet18.yaml'),
    ('cassava_resnet34',    'configs/experiments/cassava_resnet34.yaml'),
    ('cassava_resnet50',    'configs/experiments/cassava_resnet50.yaml'),
    ('cassava_vit_b16',     'configs/experiments/cassava_vit_b16.yaml'),
]

print(f"{'Model':<25s} | {'Accuracy':>8s} | {'Params':>8s} | {'Steps':>8s}")
print('-' * 65)

for name, cfg_path in pairs:
    wd = f'outputs/checkpoints/{name}'
    ckpt_tag, ckpt_path = find_best_ckpt(wd)
    if ckpt_path is None:
        print(f'{name:<25s} | {"N/A":>8s} | {"-":>8s} | {"-":>8s}')
        continue

    max_step = get_max_step(wd)
    try:
        model = build_from_exp(cfg_path)
        params = sum(p.numel() for p in model.parameters())
        acc = eval_ckpt(ckpt_path, model)
        print(f'{name:<25s} | {acc*100:7.2f}% | {params/1e6:7.1f}M | {max_step:>8,}  ({ckpt_tag})')
        del model
    except Exception as e:
        print(f'{name:<25s} | {"ERR":>8s} | {"-":>8s} | {"-":>8s}  ({e})')
        import traceback; traceback.print_exc()

# MLP from per_class_report
print(f'{"cassava_mlp":<25s} | {"63.04%":>8s} | {"~7M":>8s} | {"16,050":>8s}  (per_class_report.txt)')

# HERBS (only 10 steps)
print(f'{"cassava_herbs_swin_tiny":<25s} | {"N/A":>8s} | {"27.5M":>8s} | {"10":>8s}  (test run)')