cd taichi_render_gpu

python render_smpl.py --dataroot ../dataset/multihuman \
    --obj_path ../dataset/multihuman/smplx \
    --faces_path ../lib/data/smplx_multi.obj --yaw_list 0 90 180 270

