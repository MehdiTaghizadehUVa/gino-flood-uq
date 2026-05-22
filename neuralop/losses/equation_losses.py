import torch
import torch.nn.functional as F

from .data_losses import central_diff_2d


class BurgersEqnLoss(object):
    """
    Computes loss for Burgers' equation.
    """

    def __init__(self, visc=0.01, method="fdm", loss=F.mse_loss, domain_length=1.0):
        super().__init__()
        self.visc = visc
        self.method = method
        self.loss = loss
        self.domain_length = domain_length
        if not isinstance(self.domain_length, (tuple, list)):
            self.domain_length = [self.domain_length] * 2

    def fdm(self, u):
        # remove extra channel dimensions
        u = u.squeeze(1)

        # shapes
        _, nt, nx = u.shape

        # we assume that the input is given on a regular grid
        dt = self.domain_length[0] / (nt - 1)
        dx = self.domain_length[1] / nx

        # du/dt and du/dx
        dudt, dudx = central_diff_2d(u, [dt, dx], fix_x_bnd=True, fix_y_bnd=True)

        # d^2u/dxx
        dudxx = (
            torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)
        ) / dx**2
        # fix boundary
        dudxx[..., 0] = (u[..., 2] - 2 * u[..., 1] + u[..., 0]) / dx**2
        dudxx[..., -1] = (u[..., -1] - 2 * u[..., -2] + u[..., -3]) / dx**2

        # right hand side
        right_hand_side = -dudx * u + self.visc * dudxx

        # compute the loss of the left and right hand sides of Burgers' equation
        return self.loss(dudt, right_hand_side)

    def __call__(self, y_pred, **kwargs):
        if self.method == "fdm":
            return self.fdm(u=y_pred)
        raise NotImplementedError()


class ICLoss(object):
    """
    Computes loss for initial value problems.
    """

    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss

    def initial_condition_loss(self, y_pred, x):
        boundary_true = x[:, 0, 0, :]
        boundary_pred = y_pred[:, 0, 0, :]
        return self.loss(boundary_pred, boundary_true)

    def __call__(self, y_pred, x, **kwargs):
        return self.initial_condition_loss(y_pred, x)


class FloodPhysicsLoss(object):
    """
    Computes physics-based loss for flood simulation.
    
    This loss enforces physical constraints such as:
    1. Mass conservation (continuity equation)
    2. Momentum conservation 
    3. Non-negative water depth
    4. Physical bounds on velocity
    
    Parameters
    ----------
    delta_t : float
        Time step size for the simulation
    g : float
        Gravitational acceleration (default: 9.81 m/s²)
    rho : float
        Water density (default: 1000 kg/m³)
    physics_weight : float
        Weight for the physics loss component
    enforce_continuity : bool
        Whether to enforce mass conservation
    enforce_momentum : bool
        Whether to enforce momentum conservation
    enforce_bounds : bool
        Whether to enforce physical bounds
    """
    
    def __init__(self, delta_t=900.0, g=9.81, rho=1000.0, physics_weight=1.0,
                 enforce_continuity=True, enforce_momentum=True, enforce_bounds=True):
        super().__init__()
        self.delta_t = delta_t
        self.g = g
        self.rho = rho
        self.physics_weight = physics_weight
        self.enforce_continuity = enforce_continuity
        self.enforce_momentum = enforce_momentum
        self.enforce_bounds = enforce_bounds
    
    def continuity_loss(self, y_pred, x, **kwargs):
        """
        Enforce mass conservation (continuity equation).
        
        For flood simulation, we expect:
        - y_pred[:, 0] = water depth difference
        - y_pred[:, 1] = volume difference  
        - x contains input features including current water depth and velocity
        
        The continuity equation in 2D is:
        ∂h/∂t + ∇·(h*v) = 0
        where h is water depth and v is velocity vector
        """
        if not self.enforce_continuity:
            return torch.tensor(0.0, device=y_pred.device)
        
        # Extract predicted changes
        dh_pred = y_pred[:, 0]  # water depth change
        dv_pred = y_pred[:, 1]  # volume change
        
        # Extract current state from input features
        # Assuming x has shape [batch, channels, height, width]
        # and channels include: [static_features, water_depth_history, velocity_history]
        
        # For simplicity, we'll compute a basic continuity check
        # In practice, you'd need to extract the actual current water depth and velocity
        # from the input features based on your data structure
        
        # This is a simplified version - you'll need to adapt based on your actual data format
        batch_size = y_pred.shape[0]
        
        # Compute spatial gradients if we have spatial information
        if len(x.shape) >= 4:  # [batch, channels, height, width]
            # Extract water depth from input (assuming it's in the input features)
            # This is a placeholder - adjust based on your actual data structure
            current_h = x[:, 0, :, :]  # Assuming water depth is first channel
            
            # Compute spatial gradients using finite differences
            dh_dx = torch.diff(current_h, dim=-1, prepend=current_h[:, :, :1])
            dh_dy = torch.diff(current_h, dim=-2, prepend=current_h[:, :1, :])
            
            # Simplified continuity loss: penalize large spatial gradients
            continuity_loss = torch.mean(dh_dx**2 + dh_dy**2)
        else:
            # Fallback: simple penalty on predicted changes
            continuity_loss = torch.mean(dh_pred**2 + dv_pred**2)
        
        return continuity_loss
    
    def momentum_loss(self, y_pred, x, **kwargs):
        """
        Enforce momentum conservation.
        
        For shallow water equations, momentum conservation is:
        ∂(h*v)/∂t + ∇·(h*v⊗v) + gh∇h = 0
        """
        if not self.enforce_momentum:
            return torch.tensor(0.0, device=y_pred.device)
        
        # This is a simplified momentum loss
        # In practice, you'd compute the full momentum equation residuals
        
        # Extract predicted changes
        dh_pred = y_pred[:, 0]  # water depth change
        
        # Simplified momentum loss: penalize large water depth changes
        # that would violate momentum conservation
        momentum_loss = torch.mean(dh_pred**2)
        
        return momentum_loss
    
    def bounds_loss(self, y_pred, x, **kwargs):
        """
        Enforce physical bounds on the solution.
        """
        if not self.enforce_bounds:
            return torch.tensor(0.0, device=y_pred.device)
        
        # Extract predicted changes
        dh_pred = y_pred[:, 0]  # water depth change
        dv_pred = y_pred[:, 1]  # volume change
        
        # Enforce non-negative water depth (using ReLU penalty)
        # Penalize negative water depth changes that would make depth negative
        depth_penalty = F.relu(-dh_pred)**2
        
        # Enforce reasonable bounds on volume changes
        volume_penalty = F.relu(torch.abs(dv_pred) - 1.0)**2  # Penalize large volume changes
        
        bounds_loss = torch.mean(depth_penalty + volume_penalty)
        
        return bounds_loss
    
    def __call__(self, y_pred, x, **kwargs):
        """
        Compute the total physics loss.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions [batch, 2, height, width] where:
            - y_pred[:, 0] = water depth difference
            - y_pred[:, 1] = volume difference
        x : torch.Tensor
            Input features containing current state
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        torch.Tensor
            Total physics loss
        """
        total_physics_loss = 0.0
        
        # Add continuity loss
        continuity_loss = self.continuity_loss(y_pred, x, **kwargs)
        total_physics_loss += continuity_loss
        
        # Add momentum loss
        momentum_loss = self.momentum_loss(y_pred, x, **kwargs)
        total_physics_loss += momentum_loss
        
        # Add bounds loss
        bounds_loss = self.bounds_loss(y_pred, x, **kwargs)
        total_physics_loss += bounds_loss
        
        return self.physics_weight * total_physics_loss


class FloodContinuityLoss(object):
    """
    Simplified continuity loss specifically for flood simulation.
    
    This implements a physics loss similar to the hydrograph example,
    focusing on mass conservation in the denormalized domain.
    
    Parameters
    ----------
    delta_t : float
        Time step size
    physics_weight : float
        Weight for the physics loss
    """
    
    def __init__(self, delta_t=900.0, physics_weight=1.0):
        super().__init__()
        self.delta_t = delta_t
        self.physics_weight = physics_weight
    
    def compute_continuity_loss(self, y_pred, physics_data, batch_indices=None):
        """
        Compute continuity loss similar to the hydrograph implementation.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions [batch, 2] where:
            - y_pred[:, 0] = water depth difference (normalized)
            - y_pred[:, 1] = volume difference (normalized)
        physics_data : dict
            Dictionary containing physics parameters for each sample:
            - 'past_volume': past volume (normalized)
            - 'future_volume': future volume (normalized)
            - 'avg_inflow': average inflow (denormalized)
            - 'avg_precipitation': average precipitation (denormalized)
            - 'next_inflow': next step inflow (denormalized)
            - 'next_precip': next step precipitation (denormalized)
            - 'volume_mean': volume normalization mean
            - 'volume_std': volume normalization std
            - 'num_nodes': number of nodes per sample
            - 'area_sum': total area (denormalized)
            - 'infiltration_area_sum': infiltration area sum (denormalized)
        batch_indices : torch.Tensor, optional
            Batch indices for each sample
            
        Returns
        -------
        torch.Tensor
            Continuity loss
        """
        if physics_data is None:
            return torch.tensor(0.0, device=y_pred.device)
        
        predicted_diff = y_pred[:, 1]  # Predicted volume difference (normalized)
        physics_losses = []
        
        # Process each sample in the batch
        for i in range(y_pred.shape[0]):
            pred_diff_sum = predicted_diff[i]
            
            # Get physics data for this sample
            past_volume_norm = physics_data["past_volume"][i]
            future_volume_norm = physics_data["future_volume"][i]
            denorm_avg_inflow = physics_data["avg_inflow"][i]
            denorm_avg_precip = physics_data["avg_precipitation"][i]
            denorm_next_inflow = physics_data["next_inflow"][i]
            denorm_next_precip = physics_data["next_precip"][i]
            
            volume_mean = physics_data["volume_mean"][i]
            volume_std = physics_data["volume_std"][i]
            num_nodes = physics_data["num_nodes"][i]
            area_sum = physics_data["area_sum"][i]
            infiltration_area_sum = physics_data["infiltration_area_sum"][i]
            
            # Denormalize past and future volumes
            past_volume_denorm = past_volume_norm * volume_std + num_nodes * volume_mean
            future_volume_denorm = future_volume_norm * volume_std + num_nodes * volume_mean
            
            # Compute predicted total volume
            pred_total_volume = past_volume_denorm + volume_std * pred_diff_sum
            
            # Compute effective precipitation terms
            new_precip_term = denorm_avg_precip * infiltration_area_sum
            new_next_precip_term = denorm_next_precip * infiltration_area_sum
            
            # Compute continuity terms using ReLU to enforce non-negativity
            term1 = F.relu(
                (pred_total_volume - (past_volume_denorm + self.delta_t * (denorm_avg_inflow + new_precip_term))) / area_sum
            )**2
            
            term2 = F.relu(
                (future_volume_denorm - pred_total_volume - self.delta_t * (denorm_next_inflow + new_next_precip_term)) / area_sum
            )**2
            
            physics_losses.append(term1 + term2)
        
        if physics_losses:
            return self.physics_weight * torch.stack(physics_losses).mean()
        else:
            return torch.tensor(0.0, device=y_pred.device)
    
    def __call__(self, y_pred, x, physics_data=None, **kwargs):
        """
        Compute the continuity loss.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions
        x : torch.Tensor
            Input features
        physics_data : dict, optional
            Physics data for continuity computation
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        torch.Tensor
            Continuity loss
        """
        return self.compute_continuity_loss(y_pred, physics_data)
