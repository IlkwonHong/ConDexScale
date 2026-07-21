from calendar import c
import json
#import pickle
import os
import cv2
from manopth.manolayer import ManoLayer
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from dex_ycb_toolkit.factory import get_dataset
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh
osp = os.path

class dexycb_testfullfeed(Dataset):  
    def __init__(self, data_dir = osp.join(osp.dirname(__file__), "..", "dataset", "data"), mano_root = None, load_mesh = False, pc_sample = 1024, data_sample = None, scale_input=False, scale = None, precept = False, pred_mesh_mode="icp"):
        dexycb_dir = os.environ.get("DEX_YCB_DIR", osp.join(osp.expanduser("~"), "DexYCB"))
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        try:
            self.getdata = DexYCBDataset('s0', "test")
        except Exception as e:
            print(f"[dataset_opt] DexYCBDataset unavailable ({e}); falling back to local image paths.")
            self.getdata = None
        self.data_dir = data_dir
        self.scale_input = scale_input
        self.scale = scale
        config_file = osp.join(data_dir,"dexycb_test_s0.json")
        repo_root = osp.abspath(osp.join(osp.dirname(__file__), ".."))
        network_root = osp.join(repo_root, "network")
        network_results = osp.join(network_root, "results")
        self.hand_pred_dir = osp.join(network_results, 'hand_pose_results')
        self.obj_pred_dir = osp.join(network_results, 'obj_pose_results')
        with open(config_file, 'r') as f:
            self.meta = json.load(f)
        self.data_sample = data_sample
        if data_sample is not None:
            sample_dir = osp.join(data_dir, "test_sample_{}.json".format(data_sample))
            with open(sample_dir, 'r') as f:
                sample = json.load(f)
                all_indices = [torch.tensor(i) for i in sample.values()]
                self.sample = torch.cat(all_indices)
            self.len = len(self.sample)
        else:
            self.len = len(self.meta['images'])
            self.sample = None

        self.load_mesh = load_mesh
        if mano_root is None:
            candidates = [
                osp.join(osp.dirname(__file__), '..', 'network', 'manopth', 'mano', 'models'),
            ]
            self.mano_root = next((p for p in candidates if osp.isdir(p)), candidates[0])
        else:
            self.mano_root = mano_root
        self.precept = precept
        self.pred_mesh_mode = pred_mesh_mode
        pred_mesh_dirs = {
            "icp": ("mesh", "mesh_hand"),
            "free": ("mesh_free", "mesh_hand_free"),
        }
        if self.pred_mesh_mode not in pred_mesh_dirs:
            raise ValueError(
                "pred_mesh_mode must be one of: icp, free. "
                f"Got: {self.pred_mesh_mode}"
            )
        pred_obj_dir, pred_hand_dir = pred_mesh_dirs[self.pred_mesh_mode]
        self.pred_obj_mesh_dir = osp.join(network_results, pred_obj_dir)
        self.pred_hand_mesh_dir = osp.join(network_results, pred_hand_dir)
        self.mesh_dir = osp.join(data_dir, 'mesh_data', 'mesh_obj')
        self.point_dir = osp.join(data_dir, 'points_{}'.format(pc_sample))
        self.mano_layer = ManoLayer(
                    ncomps = 45,
                    side = 'right',
                    mano_root= self.mano_root,
                    use_pca=False,
                    flat_hand_mean=True
                    )
        self.mano_layer_pca = ManoLayer(
                    ncomps = 15,
                    side = 'right',
                    center_idx = 0,
                    mano_root= self.mano_root,
                    use_pca=True,
                    flat_hand_mean=True
                    )
        face_candidates = [
            osp.join(osp.dirname(__file__), 'closed_fmano.npy'),
            osp.join(osp.dirname(__file__), '..', 'dataset', 'data', 'closed_fmano.npy'),
        ]
        face_path = next((p for p in face_candidates if osp.exists(p)), None)
        if face_path is not None:
            self.faces = torch.LongTensor(np.load(face_path))
        else:
            self.faces = self.mano_layer.th_faces.detach().cpu().long()
    def __len__(self):
        return self.len

    def __getitem__(self, idx_):
        if self.sample is not None:
            idx = self.sample[idx_].item()
        else:
            idx = idx_
        s0_id = self.meta['images'][idx]['id']
        file_name = self.meta['images'][idx]['file_name']
        if self.getdata is not None:
            color = cv2.imread(self.getdata[s0_id]['color_file'])
        else:
            color = cv2.imread(osp.join(self.data_dir, 'images', 'test', file_name + '.png'))
        ycb_id = torch.tensor(self.meta['annotations'][idx]['ycb_id'], dtype = torch.long)
        dexycb_s0_id = torch.tensor(s0_id, dtype = torch.long)
        try:
            if self.precept:
                with open(osp.join(self.hand_pred_dir, file_name + '.json'), 'r') as f:
                    hand_pred_pose = json.load(f)
                hand_mesh_path = osp.join(self.pred_hand_mesh_dir, file_name + '_hand.ply')
                obj_mesh_path = osp.join(self.pred_obj_mesh_dir, file_name + '_obj.ply')
                if not osp.exists(hand_mesh_path) or not osp.exists(obj_mesh_path):
                    raise FileNotFoundError(
                        "Predicted meshes are missing for precept mode "
                        f"(pred_mesh_mode={self.pred_mesh_mode}, sample={file_name}). "
                        f"hand_mesh={hand_mesh_path}, obj_mesh={obj_mesh_path}"
                    )
                print(
                    f"[dexycb_testfullfeed] pred_mesh_mode={self.pred_mesh_mode} "
                    f"obj_mesh={obj_mesh_path} hand_mesh={hand_mesh_path}"
                )
                cam_extr = torch.tensor(hand_pred_pose['cam_extr'], dtype = torch.float32)
                hand_mesh = trimesh.load_mesh(hand_mesh_path)
                obj_mesh = trimesh.load_mesh(obj_mesh_path)
                obj_centre = torch.tensor(obj_mesh.center_mass, dtype = torch.float32)
                hand_joint = torch.tensor(hand_pred_pose['joints'], dtype = torch.float32)
                hand_joint = (cam_extr @ hand_joint.transpose(1,0)).transpose(1,0)
                hand_trans = torch.tensor([0.0, 0.0, 0.0], dtype = torch.float32)
                global_trans = torch.tensor(hand_pred_pose['global_trans'], dtype = torch.float32)

                new_trans = global_trans.reshape(16,4,4)[0]
                new_trans[:3,3] = hand_joint[0]

                pose = torch.eye(4, dtype = torch.float32)
                return dict(color_img=color, s0 = dexycb_s0_id, file_name = file_name), dict(verts=torch.tensor(np.array(obj_mesh.vertices), dtype=torch.float32)-hand_trans, faces=torch.tensor(np.array(obj_mesh.faces)), pose=pose, centre=obj_centre, hand_trans = hand_trans), dict(verts=torch.tensor(np.array(hand_mesh.vertices, dtype=np.float32)), faces=self.faces, joint=hand_joint, transformation=new_trans.squeeze(0))
            else:

                hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'], dtype = torch.float32)
                hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32)
                hand_joint = hand_joint - hand_trans
                obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
                mano_pose = torch.tensor(self.meta['annotations'][idx]['hand_poses'], dtype = torch.float32).view(1, -1)
                mano_shape = torch.tensor(self.meta['annotations'][idx]['hand_shapes'], dtype = torch.float32).view(1, -1)
                verts, th_jtr, th_full_pose, th_results_global, center_joint, th_trans = self.mano_layer(th_pose_coeffs = mano_pose[:,0:48], th_betas = mano_shape, th_trans=torch.tensor([0.0,0.0,0.0]), root_palm=False)  
                obj_centre = obj_centre - hand_trans


                mesh_name = self.meta['images'][idx]['file_name'] + '.obj' #already transformed
                mesh = trimesh.load_mesh(osp.join(self.mesh_dir, mesh_name))
                pose = torch.tensor(self.meta['annotations'][idx]['obj_transform'])
                return dict(color_img=color, ycb = ycb_id, s0 = dexycb_s0_id, file_name = file_name), dict(verts=torch.tensor(np.array(mesh.vertices))-hand_trans, faces=torch.tensor(np.array(mesh.faces)), pose=pose, centre=obj_centre, hand_trans = hand_trans), dict(verts=verts.squeeze(0), faces=self.faces, joint=hand_joint, transformation=th_trans.squeeze(0))

        except Exception as e:
            print(idx_, e)
            if self.precept:
                # In predicted-mesh mode, missing prediction files should be handled by caller.
                raise
            return self.__getitem__(idx_+1)


_YCB_CLASSES = {
    1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box',
    4: '005_tomato_soup_can', 5: '006_mustard_bottle', 6: '007_tuna_fish_can',
    7: '008_pudding_box', 8: '009_gelatin_box', 9: '010_potted_meat_can',
    10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
    13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block',
    17: '037_scissors', 18: '040_large_marker', 19: '051_large_clamp',
    20: '052_extra_large_clamp', 21: '061_foam_brick',
}


class dexycb_trainfeed(Dataset):
    """Train-split dataset for subject-specific GT data generation.

    Generates object meshes on-the-fly from YCB canonical models + obj_transform,
    so no pre-built mesh_obj/*.obj files are required.
    """

    # Maps DexYCB subject folder name → numeric prefix used in file names
    _SUBJECT_PREFIX = {
        '20200709-subject-01': '1',
        '20200813-subject-02': '2',
        '20200820-subject-03': '3',
        '20200903-subject-04': '4',
        '20200908-subject-05': '5',
        '20200918-subject-06': '6',
        '20200928-subject-07': '7',
        '20201002-subject-08': '8',
        '20201015-subject-09': '9',
        '20201022-subject-10': '10',
    }

    def __init__(
        self,
        subject: str = '20200709-subject-01',
        data_dir=osp.join(osp.dirname(__file__), '..', 'dataset', 'data'),
        mano_root=None,
    ):
        dexycb_dir = os.environ.get('DEX_YCB_DIR', osp.join(osp.expanduser('~'), 'DexYCB'))
        self.model_dir = osp.join(dexycb_dir, 'models')

        config_file = osp.join(data_dir, 'dexycb_train_s0.json')
        with open(config_file, 'r') as f:
            meta = json.load(f)

        # File names are prefixed with the numeric subject index (e.g. '1_20200709...')
        prefix = self._SUBJECT_PREFIX.get(subject, subject) + '_'
        self.indices = [
            i for i, img in enumerate(meta['images'])
            if img['file_name'].startswith(prefix)
        ]
        self.images = [meta['images'][i] for i in self.indices]
        self.annotations = [meta['annotations'][i] for i in self.indices]

        if mano_root is None:
            candidates = [
                osp.join(osp.dirname(__file__), '..', 'network', 'manopth', 'mano', 'models'),
            ]
            mano_root = next((p for p in candidates if osp.isdir(p)), candidates[0])

        self.mano_layer = ManoLayer(
            ncomps=45, side='right', mano_root=mano_root,
            use_pca=False, flat_hand_mean=True,
        )
        face_candidates = [
            osp.join(osp.dirname(__file__), 'closed_fmano.npy'),
            osp.join(data_dir, 'closed_fmano.npy'),
        ]
        face_path = next((p for p in face_candidates if osp.exists(p)), None)
        self.faces = (
            torch.LongTensor(np.load(face_path)) if face_path
            else self.mano_layer.th_faces.detach().cpu().long()
        )
        self.data_sample = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = self.images[idx]
        file_name = img['file_name']

        hand_trans = torch.tensor(ann['hand_trans'], dtype=torch.float32)
        hand_joint = torch.tensor(ann['hand_joints_3d'], dtype=torch.float32) - hand_trans
        obj_centre = torch.tensor(ann['obj_center_3d'], dtype=torch.float32) - hand_trans

        mano_pose = torch.tensor(ann['hand_poses'], dtype=torch.float32).view(1, -1)
        mano_shape = torch.tensor(ann['hand_shapes'], dtype=torch.float32).view(1, -1)
        verts, _, _, _, _, th_trans = self.mano_layer(
            th_pose_coeffs=mano_pose[:, :48],
            th_betas=mano_shape,
            th_trans=torch.tensor([0.0, 0.0, 0.0]),
            root_palm=False,
        )

        # Build object mesh on-the-fly from YCB canonical model + obj_transform
        ycb_id = ann['ycb_id']
        class_name = _YCB_CLASSES[ycb_id]
        mesh = trimesh.load_mesh(
            osp.join(self.model_dir, class_name, 'textured_simple.obj'),
            process=False,
        )
        T = np.array(ann['obj_transform'], dtype=np.float32)
        raw_verts = np.array(mesh.vertices, dtype=np.float32)
        obj_verts = (T[:3, :3] @ raw_verts.T).T + T[:3, 3]
        obj_verts_local = obj_verts - hand_trans.numpy()
        obj_faces = np.array(mesh.faces)

        pose = torch.tensor(ann['obj_transform'], dtype=torch.float32)
        ycb_id_t = torch.tensor(ycb_id, dtype=torch.long)
        s0_id_t = torch.tensor(img['id'], dtype=torch.long)

        input_pack = dict(color_img=None, ycb=ycb_id_t, s0=s0_id_t, file_name=file_name)
        obj_pack = dict(
            verts=torch.tensor(obj_verts_local, dtype=torch.float32),
            faces=torch.tensor(obj_faces),
            pose=pose,
            centre=obj_centre,
            hand_trans=hand_trans,
        )
        mano_pack = dict(
            verts=verts.squeeze(0),
            faces=self.faces,
            joint=hand_joint,
            transformation=th_trans.squeeze(0),
        )
        return input_pack, obj_pack, mano_pack


class ArcticFeed(Dataset):
    """Arctic bimanual dataset adapter for the genhand fc_optimization pipeline.

    Returns (input_pack, obj_pack, mano_pack) compatible with
    Optimization.run_idx_json_bimanual(). mano_pack carries both right
    (base keys) and left (_left suffix) hand data.
    """

    def __init__(
        self,
        arctic_root,
        split='train',
        protocol='p1',
        frame_stride=30,
        mano_root=None,
    ):
        self.arctic_root = arctic_root
        self.data_root = osp.join(arctic_root, 'data')
        self.frame_stride = frame_stride
        self.data_sample = None  # compat with Optimization.run_idx_json

        if mano_root is None:
            candidates = [
                osp.join(osp.dirname(__file__), '..', 'network', 'manopth', 'mano', 'models'),
            ]
            mano_root = next((p for p in candidates if osp.isdir(p)), candidates[0])

        self.mano_right = ManoLayer(
            ncomps=45, side='right', mano_root=mano_root,
            use_pca=False, flat_hand_mean=False,
        )
        self.mano_left = ManoLayer(
            ncomps=45, side='left', mano_root=mano_root,
            use_pca=False, flat_hand_mean=False,
        )

        face_candidates = [
            osp.join(osp.dirname(__file__), 'closed_fmano.npy'),
            osp.join(osp.dirname(__file__), '..', 'dataset', 'data', 'closed_fmano.npy'),
        ]
        face_path = next((p for p in face_candidates if osp.exists(p)), None)
        self.mano_faces = (
            torch.LongTensor(np.load(face_path)) if face_path
            else self.mano_right.th_faces.detach().cpu().long()
        )

        # Load canonical meshes for all objects
        vtemplate_root = osp.join(self.data_root, 'meta', 'object_vtemplates')
        self.obj_templates = {}  # {obj_name: (verts_mm, faces, parts)}
        for obj_name in os.listdir(vtemplate_root):
            mesh_path = osp.join(vtemplate_root, obj_name, 'mesh.obj')
            parts_path = osp.join(vtemplate_root, obj_name, 'parts.json')
            if not osp.exists(mesh_path) or not osp.exists(parts_path):
                continue
            mesh = trimesh.load_mesh(mesh_path, process=False)
            with open(parts_path) as f:
                parts = np.array(json.load(f), dtype=np.int32)
            self.obj_templates[obj_name] = (
                np.array(mesh.vertices, dtype=np.float32),
                np.array(mesh.faces, dtype=np.int64),
                parts,
            )

        # Build frame index from split
        split_path = osp.join(
            self.data_root, 'splits_json', f'protocol_{protocol}.json'
        )
        with open(split_path) as f:
            sequences = json.load(f)[split]  # ["s01/box_grab_01", ...]

        self.frame_index = []  # [(subject, seq_name, frame_t), ...]
        for seq_path in sequences:
            subject, seq_name = seq_path.split('/')
            obj_name = seq_name.split('_')[0]
            if obj_name not in self.obj_templates:
                continue
            obj_npy = osp.join(self.data_root, 'raw_seqs', subject, seq_name + '.object.npy')
            if not osp.exists(obj_npy):
                continue
            T = np.load(obj_npy).shape[0]
            for t in range(0, T, frame_stride):
                self.frame_index.append((subject, seq_name, t))

    def __len__(self):
        return len(self.frame_index)

    def _apply_articulation(self, verts_mm, parts, arti_angle, global_rot_aa, trans_mm):
        """Port of Arctic ObjectTensors.forward_7d_batch articulation logic.

        The movable ("top") part rotates around canonical Z-axis [0,0,-1] by
        arti_angle, then all vertices receive the global rotation + translation.

        NOTE on part indexing: Arctic builds parts_ids = parts.json + 1 and
        articulates where parts_ids == 1, i.e. where parts.json == 0.  So the
        movable part is parts == 0 (NOT parts == 1).

        Returns (N, 3) float32 in metres, world frame.
        """
        from scipy.spatial.transform import Rotation
        verts_m = verts_mm.copy() / 1000.0  # mm → m

        R_arti = Rotation.from_rotvec([0.0, 0.0, -float(arti_angle)]).as_matrix()
        top = parts == 0
        verts_m[top] = (R_arti @ verts_m[top].T).T

        R_global = Rotation.from_rotvec(global_rot_aa.astype(np.float64)).as_matrix()
        verts_world = (R_global @ verts_m.T).T + trans_mm.astype(np.float64) / 1000.0
        return verts_world.astype(np.float32)

    def _mano_forward(self, side, t, mano_data, layer):
        """MANO forward pass → world-frame (verts, joints, root_trans).

        Passes world translation directly as th_trans so manopth applies it
        to vertices, joints, and the root transform matrix in one shot.
        """
        rot   = torch.tensor(mano_data[side]['rot'][t],   dtype=torch.float32).view(1, 3)
        pose  = torch.tensor(mano_data[side]['pose'][t],  dtype=torch.float32).view(1, 45)
        shape = torch.tensor(mano_data[side]['shape'],    dtype=torch.float32).view(1, 10)
        trans = torch.tensor(mano_data[side]['trans'][t], dtype=torch.float32)  # (3,) m

        full_pose = torch.cat([rot, pose], dim=1)  # (1, 48)
        verts, joints, _, _, _, root_trans = layer(
            th_pose_coeffs=full_pose,
            th_betas=shape,
            th_trans=trans.view(1, 3),   # manopth expects (B, 3)
            root_palm=False,
        )
        return verts.squeeze(0), joints.squeeze(0), root_trans.squeeze(0)

    def __getitem__(self, idx):
        subject, seq_name, t = self.frame_index[idx]
        obj_name = seq_name.split('_')[0]

        obj_data = np.load(
            osp.join(self.data_root, 'raw_seqs', subject, seq_name + '.object.npy')
        )  # (T, 7): [arti_angle, rot_aa(3), trans_mm(3)]
        arti_angle  = float(obj_data[t, 0])
        global_rot_aa = obj_data[t, 1:4]
        trans_mm    = obj_data[t, 4:]

        verts_mm, faces, parts = self.obj_templates[obj_name]
        obj_verts  = self._apply_articulation(verts_mm, parts, arti_angle, global_rot_aa, trans_mm)
        obj_centre = torch.tensor(obj_verts.mean(axis=0), dtype=torch.float32)

        mano_data = np.load(
            osp.join(self.data_root, 'raw_seqs', subject, seq_name + '.mano.npy'),
            allow_pickle=True,
        ).item()

        rv, rj, rt = self._mano_forward('right', t, mano_data, self.mano_right)
        lv, lj, lt = self._mano_forward('left',  t, mano_data, self.mano_left)

        input_pack = dict(
            color_img=None,
            file_name=f"{subject}_{seq_name}_{t:05d}",
            obj_name=obj_name,
        )
        obj_pack = dict(
            verts=torch.tensor(obj_verts, dtype=torch.float32),
            faces=torch.tensor(faces),
            pose=torch.eye(4, dtype=torch.float32),
            centre=obj_centre,
            hand_trans=torch.zeros(3),
            arti_angle=arti_angle,
        )
        mano_pack = dict(
            verts=rv,
            faces=self.mano_faces,
            joint=rj,
            transformation=rt,
            verts_left=lv,
            joint_left=lj,
            transformation_left=lt,
        )
        return input_pack, obj_pack, mano_pack


def write_json(data, filename):
    
    def convert_tensor_to_list(data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.tolist()
        return data
    file_name = osp.join(osp.dirname(osp.abspath(__file__)), "data", filename+".json")
    with open(file_name, 'w') as f:
        json.dump(convert_tensor_to_list(data), f, indent=4)

def image_inspection(images):
    global im_id
    im_id = 0
    fig, ax = plt.subplots()
    im = images[im_id][:,:,::-1]
    im_display = ax.imshow(im)

    def update(event):
        global im_id
        if event.key == 'right':
            im_id = (im_id + 1) % len(images)
        elif event.key == 'left':
            im_id = (im_id - 1) % len(images)
        im = images[im_id][:,:,::-1]
        im_display.set_data(im)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('key_press_event', update)
    plt.show()



if __name__ == '__main__':
    dataset = dexycb_testfullfeed(
    load_mesh=True,
    pc_sample=1024,
    data_sample=None,
    precept=False,
    pred_mesh_mode="icp"
    )
    print(dataset[70])

        
