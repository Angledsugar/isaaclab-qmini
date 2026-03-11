"""Custom environment that visualizes base_link contact by changing its color to red."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class ContactVisRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with visual contact feedback on base_link.

    When base_link contacts the ground (force > threshold), the base_link mesh
    turns red. When contact ends, it reverts to the default color.
    """

    CONTACT_THRESHOLD = 1.0  # same threshold as termination config

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._contact_vis_initialized = False
        self._base_body_ids: list[int] | None = None

    def _init_contact_vis(self):
        """Lazy initialization of contact visualization (called after first step)."""
        if self._contact_vis_initialized:
            return

        try:
            from pxr import Gf, Sdf, UsdShade

            import omni.usd

            stage = omni.usd.get_context().get_stage()

            # Create red material for contact indication
            red_mat_path = "/World/Looks/ContactRedMaterial"
            if not stage.GetPrimAtPath(red_mat_path).IsValid():
                UsdShade.Material.Define(stage, red_mat_path)
                shader = UsdShade.Shader.Define(stage, red_mat_path + "/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                mat = UsdShade.Material.Get(stage, red_mat_path)
                mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            # Create default (gray) material for no-contact state
            default_mat_path = "/World/Looks/ContactDefaultMaterial"
            if not stage.GetPrimAtPath(default_mat_path).IsValid():
                UsdShade.Material.Define(stage, default_mat_path)
                shader = UsdShade.Shader.Define(stage, default_mat_path + "/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                mat = UsdShade.Material.Get(stage, default_mat_path)
                mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

            self._stage = stage
            self._red_mat_path = red_mat_path
            self._default_mat_path = default_mat_path

            # Resolve base_link body IDs from the contact sensor
            contact_sensor = self.scene.sensors["contact_forces"]
            # Find the body index for "base_link"
            body_names = contact_sensor.body_names
            self._base_body_ids = [i for i, name in enumerate(body_names) if name == "base_link"]

            if not self._base_body_ids:
                print("[ContactVisRLEnv] WARNING: 'base_link' not found in contact sensor bodies. "
                      "Contact visualization disabled.")

            # Track previous contact state to minimize USD operations
            self._prev_contact_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            self._contact_vis_initialized = True
            print(f"[ContactVisRLEnv] Contact visualization initialized for {self.num_envs} envs. "
                  f"base_link body_ids: {self._base_body_ids}")

        except Exception as e:
            print(f"[ContactVisRLEnv] Failed to initialize contact visualization: {e}")
            self._contact_vis_initialized = True  # prevent retrying
            self._base_body_ids = None

    def step(self, action: torch.Tensor):
        """Override step to add contact color visualization."""
        result = super().step(action)

        # Only update visuals if GUI is active
        if self.sim.has_gui():
            self._init_contact_vis()
            self._update_contact_colors()

        return result

    def _update_contact_colors(self):
        """Check base_link contact and update mesh colors."""
        if not self._base_body_ids:
            return

        try:
            from pxr import UsdShade

            contact_sensor = self.scene.sensors["contact_forces"]
            # net_forces_w shape: (num_envs, num_bodies, 3)
            net_forces = contact_sensor.data.net_forces_w
            # Get force magnitude for base_link bodies
            base_forces = net_forces[:, self._base_body_ids, :]
            # Check if any base body has contact force exceeding threshold
            contact_mask = torch.any(
                torch.norm(base_forces, dim=-1) > self.CONTACT_THRESHOLD, dim=-1
            )  # (num_envs,)

            # Only update envs where contact state changed
            changed = contact_mask != self._prev_contact_state
            if not changed.any():
                return

            red_mat = UsdShade.Material.Get(self._stage, self._red_mat_path)
            default_mat = UsdShade.Material.Get(self._stage, self._default_mat_path)

            changed_ids = changed.nonzero(as_tuple=False).squeeze(-1)
            for env_id in changed_ids.tolist():
                # Find the base_link prim for this environment
                base_prim_path = f"/World/envs/env_{env_id}/Robot/base_link"
                prim = self._stage.GetPrimAtPath(base_prim_path)
                if not prim.IsValid():
                    continue

                # Find mesh children to bind material to
                mesh_prims = self._find_mesh_prims(prim)
                mat = red_mat if contact_mask[env_id] else default_mat

                for mesh_prim in mesh_prims:
                    binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
                    binding_api.Bind(mat)

            self._prev_contact_state = contact_mask.clone()

        except Exception as e:
            print(f"[ContactVisRLEnv] Error updating contact colors: {e}")
            self._base_body_ids = None  # disable on error

    def _find_mesh_prims(self, prim):
        """Recursively find all Mesh prims under the given prim."""
        meshes = []
        if prim.GetTypeName() == "Mesh":
            meshes.append(prim)
        for child in prim.GetChildren():
            meshes.extend(self._find_mesh_prims(child))
        return meshes
