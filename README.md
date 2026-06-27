# Dual Latent Space Tuning for Diffusability and Semantic Alignment (Flow Leash)

Дообучение VAE-токенизатора для латентной диффузии с двумя дополнительными
целями: семантическим выравниванием латента с признаками DINOv2 и диффузионным
критиком на основе замороженной потоковой модели. Базовый автоэнкодер —
`stabilityai/sd-vae-ft-mse`, генератор для оценки — трансформер SiT.

## Установка

```
pip install -r requirements.txt
```

DINOv2 загружается через `torch.hub` при первом запуске, веса SD-VAE — через
`diffusers`.

## Данные

ImageNet в формате `ImageFolder`:

```
data/imagenet/train/<class>/*.JPEG
data/imagenet/val/<class>/*.JPEG
```

Путь задаётся параметром `data.root` в конфиге или через `-s data.root=...`.

## Запуск

Параметры берутся из YAML-конфига (`-c`), отдельные значения можно
переопределить флагом `-s ключ=значение`.

1. Закодировать латенты исходного автоэнкодера:

```
python encode_latents.py -c configs/base.yaml \
  -s data.root=data/imagenet/train io.out_dir=runs/latents/base
```

2. Обучить критик на этих латентах:

```
python train_gen.py -c configs/critic.yaml
```

3. Обучить три варианта токенизатора:

```
python train_vae.py -c configs/vae_base.yaml -s data.root=data/imagenet/train
python train_vae.py -c configs/vae_vfm.yaml  -s data.root=data/imagenet/train
python train_vae.py -c configs/vae_ours.yaml -s data.root=data/imagenet/train
```

4. Для каждого варианта закодировать латенты и обучить генератор
   (пример для `ours`, для остальных меняются пути):

```
python encode_latents.py -c configs/base.yaml \
  -s data.root=data/imagenet/train vae.checkpoint=runs/vae_ours/vae.pt \
     io.out_dir=runs/latents/ours
python train_gen.py -c configs/gen.yaml \
  -s data.latent_dir=runs/latents/ours io.out_dir=runs/gen_ours
```

5. Посчитать метрики:

```
python eval_fid.py -c configs/eval.yaml \
  -s data.root=data/imagenet/val \
     eval.vae_ckpt=runs/vae_ours/vae.pt eval.gen_ckpt=runs/gen_ours/model.pt
python eval_recon.py -c configs/eval.yaml \
  -s data.root=data/imagenet/val eval.vae_ckpt=runs/vae_ours/vae.pt
```

`eval_fid.py` печатает gFID, `eval_recon.py` — PSNR, LPIPS, rFID и статистику
латента. Результаты сохраняются в JSON рядом с чекпойнтами.

## Конфиги

- `configs/base.yaml` — общие параметры (данные, VAE, лоссы, оптимизация);
- `configs/vae_base.yaml`, `vae_vfm.yaml`, `vae_ours.yaml` — три варианта
  токенизатора (отличаются весами `vfm_weight` и `critic_weight`);
- `configs/critic.yaml` — обучение критика;
- `configs/gen.yaml` — обучение генератора для оценки;
- `configs/eval.yaml` — параметры оценки.

## Структура

```
config.py          загрузка YAML-конфигов
data.py            датасеты изображений и латентов
models.py          обёртка SD-VAE и DINOv2
sit.py             трансформер SiT
flow.py            flow matching и сэмплер
losses.py          реконструкция, выравнивание, критик
metrics.py         FID и статистика латента
encode_latents.py  кодирование датасета в латенты
train_gen.py       обучение SiT (критик и генератор)
train_vae.py       дообучение токенизатора
eval_fid.py        генеративный FID
eval_recon.py      метрики восстановления
```
