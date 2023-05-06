# def calc_cov_and_corr(x: T, shift: int = -1) -> float:
#     x_shifted = tr.roll(x, shift, dims=-1)
#     both = tr.stack([x, x_shifted], dim=0)
#     corr = tr.corrcoef(both)
#     return corr[0, 1].item()
#
#
# def calc_auto_corr(x: T) -> T:
#     var = x.var()
#     std = x.std()
#     padding = tr.zeros_like(x)
#     padding = padding[..., :-1]
#     # padding = padding[..., :8]
#     x_in = tr.cat([padding, x], dim=-1)
#     # x_in = tr.clone(x)
#     x = x.view(1, 1, -1)
#     x_in = x_in.view(1, 1, -1)
#     # corr = nn.functional.conv1d(x_in, x, padding=x.size(-1) - 1)
#     corr = nn.functional.conv1d(x_in, x, padding=0) / (x ** 2).sum()
#     return corr.abs()
#
#
# def calc_stft(signal: T) -> T:
#     stft = tr.stft(
#         signal,
#         n_fft=32,
#         center=False,
#         return_complex=True,
#         hop_length=1,
#         normalized=False,
#     ).abs().squeeze(1)
#     stft += 1e-8
#     # stft += 1.0
#     stft = tr.log(stft)
#     # stft = stft.prod(dim=1).unsqueeze(-1)
#     return stft
#
#
# def calc_normalized_entropy(x: T, eps: float = 1e-8) -> (T, T):
#     assert x.ndim >= 2
#     x_min, _ = x.min(dim=-2)
#     x -= x_min
#     x += eps
#     x_sum = x.sum(dim=-2)
#     x /= x_sum
#     log_x = tr.log(x)
#     entrop = (-x * log_x).sum(dim=-2)
#     entrop /= tr.log(tr.tensor(x.size(0)))
#     return entrop, x
#
#
# def calc_stats(mod_sig: T, sr: float) -> None:
#     print("============================================================")
#     dev_1 = mod_sig[1:] - mod_sig[:-1]
#     x = (mod_sig - 0.5) * 2.0
#     # x = dev_1
#     plt.plot(x)
#     plt.show()
#     stft = calc_stft(x)
#     # entrop, stft = calc_normalized_entropy(stft)
#     entrop, stft = calc_normalized_entropy(mod_sig.view(-1, 1))
#     plt.imshow(stft, interpolation="none", aspect="auto")
#     plt.title(f"mean entropy = {entrop.mean():.4f}")
#     plt.show()
#
#
# if __name__ == "__main__":
#     # a = tr.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     # b = tr.tensor([0.7, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
#     # c = tr.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.7])
#     # d = tr.tensor([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
#     # e = tr.tensor([0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0])
#     # f = tr.tensor([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     # print(calc_entropy(a))
#     # print(calc_entropy(b))
#     # print(calc_entropy(c))
#     # print(calc_entropy(d))
#     # print(calc_entropy(e))
#     # print(calc_entropy(f))
#     # exit()
#
#     sr = 169
#     m_h = tr.load("../out/m_h_1.pt")
#     plt.plot(m_h)
#     plt.show()
#     calc_stats(m_h, sr)
#     plt.show()
#     # sr = 44100
#     n_samples = m_h.size(-1)
#     shape = "tri"
#     mod_sig = make_mod_signal(n_samples, sr, 3.0, phase=0, shape=shape, exp=1.0)
#     # plt.plot(mod_sig)
#     # plt.show()
#     calc_stats(mod_sig, sr)
#     mod_sig = make_mod_signal(n_samples, sr, 1.0, phase=0, shape=shape, exp=1.0)
#     # plt.plot(mod_sig)
#     # plt.show()
#     calc_stats(mod_sig, sr)
#     mod_sig = make_mod_signal(n_samples, sr, 0.1, phase=0, shape=shape, exp=1.0)
#     # plt.plot(mod_sig)
#     # plt.show()
#     calc_stats(mod_sig, sr)
#     # noise = tr.rand_like(mod_sig)
#     # plt.plot(noise)
#     # plt.show()
#     # calc_stats(noise, sr)
#     # line = tr.ones_like(mod_sig) / 2.0
#     # plt.plot(line)
#     # plt.show()
#     # calc_stats(line, sr)
#     exit()


# =========================================================================================


# if is_training:
#     # save_dir = "../data/idmt_4_fl_all_2/train"
#     # save_dir = "../data/idmt_4_fl_all_2_quasi/train"
#     # save_dir = "../data/idmt_4_ch_all_2/train"
#     # save_dir = "../data/idmt_4_ch_all_2_quasi/train"
#     # save_dir = "../data/egfx_clean_44100_fl_all_2/train"
#     # save_dir = "../data/egfx_clean_44100_ch_all_2/train"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/idmt_4_fl_all_2/train"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/idmt_4_fl_all_2_quasi/train"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/idmt_4_ch_all_2/train"
#     save_dir = "/import/c4dm-datasets-ext/cm007/idmt_4_ch_all_2_quasi/train"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/egfx_clean_44100_fl_all_2/train"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/egfx_clean_44100_ch_all_2/train"
# else:
#     # save_dir = "../data/idmt_4_fl_all_2/val"
#     # save_dir = "../data/idmt_4_fl_all_2_quasi/val"
#     # save_dir = "../data/idmt_4_ch_all_2/val"
#     # save_dir = "../data/idmt_4_ch_all_2_quasi/val"
#     # save_dir = "../data/egfx_clean_44100_fl_all_2/val"
#     # save_dir = "../data/egfx_clean_44100_ch_all_2/val"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/idmt_4_fl_all_2/val"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/idmt_4_fl_all_2_quasi/val"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/idmt_4_ch_all_2/val"
#     save_dir = "/import/c4dm-datasets-ext/cm007/idmt_4_ch_all_2_quasi/val"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/egfx_clean_44100_fl_all_2/val"
#     # save_dir = "/import/c4dm-datasets-ext/cm007/egfx_clean_44100_ch_all_2/val"
# os.makedirs(save_dir, exist_ok=True)
# assert os.path.isdir(save_dir)
# for idx in range(dry.size(0)):
#     d = dry[idx]
#
#     w = wet[idx]
#     m = mod_sig[idx]
#     m = linear_interpolate_last_dim(m, m.size(-1) // 100, align_corners=True)
#     f = {k: v if isinstance(v, float) else v[idx] for k, v in fx_params.items()}
#     f = {k: v.item() if isinstance(v, T) else v for k, v in f.items()}
#     save_dict = {
#         "mod_sig": m,
#         "fx_params": f,
#     }
#
#     hash_dict = {k: str(v.numpy()) if isinstance(v, T) else str(v) for k, v in f.items()}
#     data_md5 = hashlib.md5(json.dumps(hash_dict, sort_keys=True).encode('utf-8')).hexdigest()
#     tr.save(save_dict, os.path.join(save_dir, f"{data_md5}.pt"))
#     torchaudio.save(os.path.join(save_dir, f"{data_md5}_dry.wav"), d, int(self.sr))
#     torchaudio.save(os.path.join(save_dir, f"{data_md5}_wet.wav"), w, int(self.sr))


# =========================================================================================


# dry = wet
# freq_all = fx_params["rate_hz"]
# freq_all *= (1.0 + ((tr.rand_like(freq_all) - 0.5) * 2.0) * 0.25)
# freq_all = tr.clip(freq_all, 0.5, 3.0)
#
# phase_all = fx_params["phase"]
# phase_all += (((tr.rand_like(phase_all) - 0.5) * 2.0) * tr.pi * 0.5)
# phase_all = (phase_all + (2 * tr.pi)) % (2 * tr.pi)
#
# mod_sig_hat = make_rand_mod_signal(
#     wet.size(0),
#     345,
#     sr=172.5,
#     freq_min=0.5,
#     freq_max=3.0,
#     # shapes=["cos"],
#     # shapes=["tri"],
#     # shapes=["rect_cos"],
#     # shapes=["inv_rect_cos"],
#     # shapes=["saw"],
#     # shapes=["rsaw"],
#     shapes=None,
#     # phase_all=phase_all,
#     # freq_all=freq_all,
# )
# # mod_sig_hat = tr.rand((wet.size(0), 345))


# dry = wet
# phase_all = fx_params["phase"]
# freq_all = fx_params["rate_hz"]
# shapes_all = fx_params["shape"]
# mod_sig_hat = make_rand_mod_signal(
#     batch_size=wet.size(0),
#     n_samples=345,
#     sr=172.5,
#     freq_min=0.5,
#     freq_max=3.0,
#     shapes_all=shapes_all,
#     freq_all=freq_all,
#     phase_all=phase_all,
#     freq_error=0.25,
#     phase_error=0.5,
# )


# valid_indices = find_valid_mod_sig_indices(mod_sig_hat)
# if not valid_indices:
#     log.info("No valid LFO signals found")
#     return None, None, None
# log.info(f"Found {len(valid_indices)} valid LFO signals")
# dry = dry[valid_indices, ...]
# wet = wet[valid_indices, ...]
# mod_sig_hat = mod_sig_hat[valid_indices, ...]
# if mod_sig is not None:
#     mod_sig = mod_sig[valid_indices, ...]


# freq_all = tr.ones((wet.size(0),)) * 0.75
# freq_all = tr.ones((wet.size(0),)) * 2.0
# mod_sig_hat = make_rand_mod_signal(
#     batch_size=wet.size(0),
#     n_samples=345,
#     sr=172.5,
#     freq_min=0.5,
#     freq_max=2.0,
#     shapes=["tri"],
#     freq_all=freq_all,
#     freq_error=0.0,
#     # freq_error=0.25,
# )
# mod_sig_hat = mod_sig_hat.to(self.device)

# mod_sig_hat = tr.zeros((wet.size(0), 345)).to(self.device)
