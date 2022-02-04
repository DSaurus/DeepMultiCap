cd taichi_render_gpu
python render_multi.py --data_root ../assets/obj --texture_root ../assets/tex --save_path ../dataset/example
python render_smpl.py --dataroot ../dataset/example --obj_path ../assets/smplx --faces_path ../lib/data/smplx_fine.obj
