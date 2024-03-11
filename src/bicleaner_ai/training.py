from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
import logging
import os
import fasttext

try:
    from . import __version__
except (SystemError, ImportError):
    from bicleaner_ai import __version__


# Porn removal classifier
# training, compressing, run tests and save model file
def train_porn_removal(args):
    if args.porn_removal_train is None or args.porn_removal_file is None:
        return

    logging.info("Training porn removal classifier.")
    model = fasttext.train_supervised(args.porn_removal_train.name,
                                    thread=args.processes,
                                    lr=1.0,
                                    epoch=25,
                                    minCount=5,
                                    wordNgrams=1,
                                    verbose=0)
    logging.info("Compressing classifier.")
    model.quantize(args.porn_removal_train.name,
                retrain=True,
                thread=args.processes,
                verbose=0)

    if args.porn_removal_test is not None:
        N, p, r = model.test(args.porn_removal_test.name, threshold=0.5)
        logging.info("Precision:\t{:.3f}".format(p))
        logging.info("Recall:\t{:.3f}".format(r))

    logging.info("Saving porn removal classifier.")
    model.save_model(args.porn_removal_file)


def repr_right(numeric_list, numeric_fmt = "{:1.4f}"):
    result_str = ["["]
    for i in range(len(numeric_list)):
        result_str.append(numeric_fmt.format(numeric_list[i]))
        if i < (len(numeric_list)-1):
            result_str.append(", ")
        else:
            result_str.append("]")
    return "".join(result_str)

# Check if a file path is relative to a path
def check_relative_path(path, filepath):
    file_abs = os.path.abspath(filepath)
    path_abs = os.path.abspath(path.rstrip('/')) # remove trailing / for safety
    return file_abs.replace(path_abs + '/', '').count('/') == 0

# Write YAML with the training parameters and quality estimates
def write_metadata(args, classifier, y_true, y_pred, lm_stats):
    out = args.metadata

    # write current bicleaner ai version
    out.write(f"bicleaner_ai_version: {__version__}\n")

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    out.write(f"precision_score: {precision:.3f}\n")
    out.write(f"recall_score: {recall:.3f}\n")
    out.write(f"f1_score: {f1:.3f}\n")
    out.write(f"matthews_corr_coef: {mcc:.3f}\n")

    # Writing it by hand (not using YAML libraries) to preserve the order
    out.write(f"source_lang: {args.source_lang}\n")
    out.write(f"target_lang: {args.target_lang}\n")

    if args.porn_removal_file is not None and args.porn_removal_train is not None:
        # Save base names only if directories are relative
        if check_relative_path(args.model_dir, args.porn_removal_file):
            porn_removal_file = os.path.basename(args.porn_removal_file)
        else:
            porn_removal_file = args.porn_removal_file
        out.write(f"porn_removal_file: {porn_removal_file}\n")
        out.write(f"porn_removal_side: {args.porn_removal_side}\n")

    if lm_stats is not None and args.lm_file_sl is not None and args.lm_file_tl is not None:
        # Save base names only if directories are relative
        if check_relative_path(args.model_dir, args.lm_file_sl):
            lm_file_sl = os.path.basename(args.lm_file_sl)
        else:
            lm_file_sl = args.lm_file_sl
        if check_relative_path(args.model_dir, args.lm_file_tl):
            lm_file_tl = os.path.basename(args.lm_file_tl)
        else:
            lm_file_tl = args.lm_file_tl

        out.write(f"source_lm: {lm_file_sl}\n")
        out.write(f"target_lm: {lm_file_tl}\n")
        out.write(f"lm_type: CHARACTER\n")
        out.write(f"clean_mean_perp: {lm_stats.clean_mean}\n")
        out.write(f"clean_stddev_perp: {lm_stats.clean_stddev}\n")
        out.write(f"noisy_mean_perp: {lm_stats.noisy_mean}\n")
        out.write(f"noisy_stddev_perp: {lm_stats.noisy_stddev}\n")

    # Save classifier
    out.write(f"classifier_type: {args.classifier_type}\n")

    # Save classifier train settings
    out.write("classifier_settings:\n")
    for key in sorted(classifier.settings.keys()):
        # Don't print objects
        if type(classifier.settings[key]) in [int, str, list, tuple]:
            if type(classifier.settings[key]) in [list, tuple]:
                out.write("    " + key + ": " + repr_right(classifier.settings[key], "{:.8f}") + "\n")
            else:
                out.write("    " + key + ": " + str(classifier.settings[key]) + "\n")
