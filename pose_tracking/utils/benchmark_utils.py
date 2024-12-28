import time

import torch


@torch.no_grad()
def measure_average_inference_time(model, inputs, num_iters=100, warm_iters=5):
    ts = []
    for iter_ in range(num_iters):
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        model(inputs)
        torch.cuda.synchronize()
        t = time.perf_counter() - t_
        if iter_ >= warm_iters:
            ts.append(t)
    print(ts)
    return sum(ts) / len(ts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path")
    args, _ = parser.parse_known_args()
    ckpt_path = args.ckpt_path

    model = ...
    model.cuda()
    model.eval()
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt["model"])
    inputs = ...
    t = measure_average_inference_time(model, inputs, args.num_iters, args.warm_iters)
    fps = 1.0 / t * args.batch_size
    print(f"{fps=}")
