SAMPLE_RATE = 16_000

data = dict(
    root='wake_words/data',
    pos_prefix='pos_all',
    neg_prefix='neg',
    noise_prefix='noise',
    sample_rate=SAMPLE_RATE,
    batch_size=4,
    train_ratio=0.75
)

model = dict(
    sample_rate=SAMPLE_RATE,
    n_mels=128,
    n_fft=400,
    hidden_dim=64,
    dropout=0.1
)

trainer = dict(
    epochs=10,
    lr=1e-3,
    weight_decay=1e-3,
    ckpt_dir='wake_words/checkpoints/'
)
