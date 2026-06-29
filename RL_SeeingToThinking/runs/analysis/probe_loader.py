"""
probe_loader.py — unified probe-set loading (babyVision OR DOCCI) for all analyses.

Both return list[MCItem], so mc_eval / module_graft / depth_probe are dataset-agnostic.
"""


def add_probe_args(ap):
    ap.add_argument("--dataset", choices=["babyvision", "docci"], default="docci",
                    help="Which probe set. Default docci (in-distribution, where the RL gain lives).")
    # babyvision
    ap.add_argument("--data-dir", default=None, help="babyvision dir (meta_data.jsonl + images/)")
    # docci
    ap.add_argument("--jsonl", default=None, help="docci perception jsonl")
    ap.add_argument("--image-dir", default=None, help="docci image root")
    ap.add_argument("--n-sample", type=int, default=None, help="docci: subsample N items")
    ap.add_argument("--seed", type=int, default=1)


def load_probe(args):
    if args.dataset == "babyvision":
        from babyvision_data import load_mc_items
        assert args.data_dir, "--data-dir required for babyvision"
        return load_mc_items(args.data_dir)
    elif args.dataset == "docci":
        from docci_data import load_docci_items
        assert args.jsonl and args.image_dir, "--jsonl and --image-dir required for docci"
        return load_docci_items(args.jsonl, args.image_dir, n_sample=args.n_sample, seed=args.seed)
    raise ValueError(args.dataset)
