"""Custom environment that visualizes ground contact by changing body colors to red."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


class ContactVisRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with visual contact feedback on all robot bodies.

    When any robot body contacts the ground (force > threshold), that body's mesh
    turns red. When contact ends, it reverts to the default color.
    """

    CONTACT_THRESHOLD = 1.0  # Newtons

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._contact_vis_initialized = False
        self._body_names: list[str] | None = None

    def _create_omnipbr_material(self, stage, mat_path: str, color: tuple):
        """Create an OmniPBR material compatible with Isaac Sim's RTX renderer."""
        from pxr import Gf, Sdf, UsdShade

        if stage.GetPrimAtPath(mat_path).IsValid():
            return

        UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shader.CreateIdAttr("OmniPBR")
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.5)

        mat = UsdShade.Material.Get(stage, mat_path)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

    def _init_contact_vis(self):
        """Lazy initialization of contact visualization (called after first step)."""
        if self._contact_vis_initialized:
            return

        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()

            # Create OmniPBR materials (RTX renderer compatible)
            red_mat_path = "/World/Looks/ContactRedMaterial"
            default_mat_path = "/World/Looks/ContactDefaultMaterial"
            self._create_omnipbr_material(stage, red_mat_path, (1.0, 0.0, 0.0))
            self._create_omnipbr_material(stage, default_mat_path, (0.8, 0.8, 0.8))

            self._stage = stage
            self._red_mat_path = red_mat_path
            self._default_mat_path = default_mat_path

            # Get all body names from the contact sensor
            contact_sensor = self.scene.sensors["contact_forces"]
            self._body_names = list(contact_sensor.body_names)
            self._num_bodies = len(self._body_names)

            if not self._body_names:
                print("[ContactVisRLEnv] WARNING: No bodies found in contact sensor. "
                      "Contact visualization disabled.")

            # Build mapping: body_name -> list of visual prim paths (from env_0)
            robot_prim_path = "/World/envs/env_0/Robot"
            self._body_visual_paths: dict[str, list[str]] = {}

            for body_name in self._body_names:
                # Isaac Sim URDF import: visuals are under {body_name}/visuals
                visuals_path = f"{robot_prim_path}/{body_name}/visuals"
                prim = stage.GetPrimAtPath(visuals_path)
                if prim.IsValid():
                    self._body_visual_paths[body_name] = [visuals_path]
                else:
                    # Fallback: body prim itself
                    body_path = f"{robot_prim_path}/{body_name}"
                    prim = stage.GetPrimAtPath(body_path)
                    if prim.IsValid():
                        self._body_visual_paths[body_name] = [body_path]
                    else:
                        self._body_visual_paths[body_name] = []

            # Store the original material binding for each body (to restore later)
            self._body_original_mat: dict[str, str | None] = {}
            for body_name, paths in self._body_visual_paths.items():
                from pxr import UsdShade
                if paths:
                    prim = stage.GetPrimAtPath(paths[0])
                    if prim.IsValid():
                        binding_api = UsdShade.MaterialBindingAPI(prim)
                        bound_mat, _ = binding_api.ComputeBoundMaterial()
                        if bound_mat:
                            self._body_original_mat[body_name] = bound_mat.GetPath().pathString
                        else:
                            self._body_original_mat[body_name] = default_mat_path
                    else:
                        self._body_original_mat[body_name] = default_mat_path
                else:
                    self._body_original_mat[body_name] = default_mat_path

            # Track previous contact state per body
            # Shape: (num_envs, num_bodies)
            self._prev_contact_state = torch.zeros(
                self.num_envs, self._num_bodies, dtype=torch.bool, device=self.device
            )

            self._contact_vis_initialized = True

            mapped = [n for n, p in self._body_visual_paths.items() if p]
            print(f"[ContactVisRLEnv] Contact visualization initialized: "
                  f"{len(mapped)}/{self._num_bodies} bodies mapped")

        except Exception as e:
            import traceback
            print(f"[ContactVisRLEnv] Failed to initialize contact visualization: {e}")
            traceback.print_exc()
            self._contact_vis_initialized = True
            self._body_names = None

    def _reset_contact_colors(self, env_ids: list[int]):
        """Restore original material colors for reset environments."""
        if not self._contact_vis_initialized or not self._body_names:
            return

        try:
            from pxr import UsdShade

            for env_id in env_ids:
                for body_idx, body_name in enumerate(self._body_names):
                    if not self._prev_contact_state[env_id, body_idx]:
                        continue

                    template_paths = self._body_visual_paths.get(body_name, [])
                    orig_path = self._body_original_mat.get(body_name, self._default_mat_path)
                    mat = UsdShade.Material.Get(self._stage, orig_path)

                    for tmpl_path in template_paths:
                        prim_path = tmpl_path.replace("/env_0/", f"/env_{env_id}/")
                        prim = self._stage.GetPrimAtPath(prim_path)
                        if prim.IsValid():
                            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                            binding_api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)

                    self._prev_contact_state[env_id, body_idx] = False

        except Exception as e:
            print(f"[ContactVisRLEnv] Error resetting contact colors: {e}")

    def step(self, action: torch.Tensor):
        """Override step to add contact color visualization."""
        result = super().step(action)

        if self.sim.has_gui():
            self._init_contact_vis()

            # Restore colors for environments that were just reset
            reset_env_ids = (self.episode_length_buf == 1).nonzero(as_tuple=False).squeeze(-1).tolist()
            if reset_env_ids:
                self._reset_contact_colors(reset_env_ids)

            self._update_contact_colors()

        return result

    def _update_contact_colors(self):
        """Check all body contacts and update mesh colors."""
        if not self._body_names:
            return

        try:
            from pxr import UsdShade

            contact_sensor = self.scene.sensors["contact_forces"]
            net_forces = contact_sensor.data.net_forces_w

            force_magnitudes = torch.norm(net_forces, dim=-1)
            contact_mask = force_magnitudes > self.CONTACT_THRESHOLD

            # Log newly contacted bodies
            new_contacts = contact_mask & ~self._prev_contact_state
            if new_contacts.any():
                for env_id in range(self.num_envs):
                    body_ids = new_contacts[env_id].nonzero(as_tuple=False).squeeze(-1).tolist()
                    if body_ids:
                        names = [self._body_names[i] for i in body_ids]
                        forces = [f"{force_magnitudes[env_id, i].item():.1f}N" for i in body_ids]
                        contact_info = ", ".join(f"{n}({f})" for n, f in zip(names, forces))
                        print(f"[Contact] env_{env_id}: {contact_info}")

            changed = contact_mask != self._prev_contact_state
            if not changed.any():
                return

            red_mat = UsdShade.Material.Get(self._stage, self._red_mat_path)

            env_ids, body_ids = changed.nonzero(as_tuple=True)

            for env_id, body_id in zip(env_ids.tolist(), body_ids.tolist()):
                body_name = self._body_names[body_id]
                template_paths = self._body_visual_paths.get(body_name, [])
                if not template_paths:
                    continue

                is_contact = contact_mask[env_id, body_id].item()

                if is_contact:
                    mat = red_mat
                else:
                    # Restore original material
                    orig_path = self._body_original_mat.get(body_name, self._default_mat_path)
                    mat = UsdShade.Material.Get(self._stage, orig_path)

                for tmpl_path in template_paths:
                    prim_path = tmpl_path.replace("/env_0/", f"/env_{env_id}/")
                    prim = self._stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                        binding_api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)

            self._prev_contact_state = contact_mask.clone()

        except Exception as e:
            import traceback
            print(f"[ContactVisRLEnv] Error updating contact colors: {e}")
            traceback.print_exc()
            self._body_names = None
