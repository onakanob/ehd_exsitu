from pic_parser.patches_gui import run_alignment_gui


if __name__ == "__main__":
        app = run_alignment_gui(im_path="./tests/patch_test_files/2-may-22 run1.bmp",
                            pattern_path="./tests/patch_test_files/pattern.txt",
                            params_file="./tests/patch_test_files/pattern_params.json")
