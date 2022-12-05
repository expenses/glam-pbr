#![no_std]

use core::{
    f32::consts::{FRAC_1_PI, PI},
    ops::Add,
};
use glam::{Mat4, Vec2, Vec3, Vec4Swizzles};
use num_traits::float::Float;

// Workarounds: can't use f32.lerp, f32.clamp or f32.powi.

// https://docs.gl/sl4/reflect
pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - 2.0 * normal.dot(incident) * normal
}

pub fn light_direction_and_attenuation(
    fragment_position: Vec3,
    light_position: Vec3,
) -> (Vec3, f32, f32) {
    let vector = light_position - fragment_position;
    let distance_sq = vector.length_squared();
    let distance = distance_sq.sqrt();
    let direction = vector / distance;
    let attenuation = 1.0 / distance_sq;

    (direction, distance, attenuation)
}

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

#[derive(Copy, Clone)]
pub struct View(pub Vec3);

impl ShadingVector for View {
    fn vector(&self) -> Vec3 {
        self.0
    }
}

#[derive(Copy, Clone)]
pub struct Light(pub Vec3);

impl ShadingVector for Light {
    fn vector(&self) -> Vec3 {
        self.0
    }
}

/// A vector used in shading. It is important that the vector is normalised and points away from the surface of the object being shaded.
pub trait ShadingVector {
    fn vector(&self) -> Vec3;
}

#[derive(Copy, Clone)]
pub struct Normal(pub Vec3);

impl ShadingVector for Normal {
    fn vector(&self) -> Vec3 {
        self.0
    }
}

#[derive(Copy, Clone)]
pub struct Halfway(Vec3);

impl Halfway {
    pub fn new(view: &View, light: &Light) -> Self {
        Self((view.0 + light.0).normalize())
    }
}

impl ShadingVector for Halfway {
    fn vector(&self) -> Vec3 {
        self.0
    }
}

pub struct Dot<A, B> {
    pub value: f32,
    _phantom: core::marker::PhantomData<(A, B)>,
}

impl<A, B> Clone for Dot<A, B> {
    fn clone(&self) -> Self {
        Self {
            value: self.value,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<A, B> Copy for Dot<A, B> {}

impl<A: ShadingVector, B: ShadingVector> Dot<A, B> {
    pub fn new(a: &A, b: &B) -> Self {
        Self {
            value: a.vector().dot(b.vector()).max(core::f32::EPSILON),
            _phantom: core::marker::PhantomData,
        }
    }
}

pub fn d_ggx(normal_dot_halfway: Dot<Normal, Halfway>, roughness: ActualRoughness) -> f32 {
    let noh = normal_dot_halfway.value;

    let alpha_roughness_sq = roughness.0 * roughness.0;

    let f = (noh * noh) * (alpha_roughness_sq - 1.0) + 1.0;

    alpha_roughness_sq / (PI * f * f)
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
// Geometric shadowing function

pub fn v_smith_ggx_correlated(
    normal_dot_view: Dot<Normal, View>,
    normal_dot_light: Dot<Normal, Light>,
    roughness: ActualRoughness,
) -> f32 {
    let nov = normal_dot_view.value;
    let nol = normal_dot_light.value;

    let a2 = roughness.0 * roughness.0;
    let ggx_v = nol * (nov * nov * (1.0 - a2) + a2).sqrt();
    let ggx_l = nov * (nol * nol * (1.0 - a2) + a2).sqrt();

    let ggx = ggx_v + ggx_l;

    if ggx > 0.0 {
        0.5 / ggx
    } else {
        0.0
    }
}

// Fresnel

pub fn fresnel_schlick(view_dot_halfway: Dot<View, Halfway>, f0: Vec3, f90: Vec3) -> Vec3 {
    f0 + (f90 - f0) * (1.0 - view_dot_halfway.value).powf(5.0)
}

#[derive(Copy, Clone)]
pub struct ActualRoughness(f32);

impl ActualRoughness {
    fn apply_ior(self, ior: IndexOfRefraction) -> ActualRoughness {
        ActualRoughness(self.0 * clamp(ior.0 * 2.0 - 2.0, 0.0, 1.0))
    }
}

#[derive(Clone, Copy)]
pub struct PerceptualRoughness(pub f32);

impl PerceptualRoughness {
    pub fn as_actual_roughness(&self) -> ActualRoughness {
        ActualRoughness(self.0 * self.0)
    }

    fn apply_ior(self, ior: IndexOfRefraction) -> PerceptualRoughness {
        PerceptualRoughness(self.0 * clamp(ior.0 * 2.0 - 2.0, 0.0, 1.0))
    }
}

pub struct BasicBrdfParams {
    pub normal: Normal,
    pub light: Light,
    pub light_intensity: Vec3,
    pub view: View,
    pub material_params: MaterialParams,
}

#[derive(Clone, Copy)]
pub struct MaterialParams {
    pub albedo_colour: Vec3,
    pub metallic: f32,
    pub perceptual_roughness: PerceptualRoughness,
    pub index_of_refraction: IndexOfRefraction,
    pub specular_colour: Vec3,
    pub specular_factor: f32,
}

impl MaterialParams {
    pub fn diffuse_colour(&self) -> Vec3 {
        // Basically the same as c_diff?
        self.albedo_colour
            * (1.0 - self.index_of_refraction.to_dielectric_f0())
            * (1.0 - self.metallic)
    }
}

#[derive(Clone, Copy)]
pub struct IndexOfRefraction(pub f32);

/// Corresponds a f0 of 4% reflectance on dielectrics ((1.0 - ior) / (1.0 + ior)) ^ 2.
impl Default for IndexOfRefraction {
    fn default() -> Self {
        Self(1.5)
    }
}

impl IndexOfRefraction {
    pub fn to_dielectric_f0(&self) -> f32 {
        let root = (self.0 - Self::AIR.0) / (self.0 + Self::AIR.0);
        root * root
    }

    const AIR: Self = Self(1.0);
}

pub fn transmission_btdf(
    material_params: MaterialParams,
    normal: Normal,
    view: View,
    light: Light,
) -> Vec3 {
    let actual_roughness = material_params.perceptual_roughness.as_actual_roughness();
    let index_of_refraction = material_params.index_of_refraction;

    let transmission_roughness = actual_roughness.apply_ior(index_of_refraction);

    let light_mirrored = Light((light.0 + 2.0 * normal.0 * (-light.0).dot(normal.0)).normalize());

    let halfway = Halfway::new(&view, &light_mirrored);
    let normal_dot_halfway = Dot::new(&normal, &halfway);
    let view_dot_halfway = Dot::new(&view, &halfway);
    let normal_dot_view = Dot::new(&normal, &view);
    let normal_dot_light_mirrored = Dot::new(&normal, &light_mirrored);

    let distribution = d_ggx(normal_dot_halfway, transmission_roughness);

    let geometric_shadowing = v_smith_ggx_correlated(
        normal_dot_view,
        normal_dot_light_mirrored,
        transmission_roughness,
    );

    let f0 = calculate_combined_f0(material_params);
    let f90 = calculate_combined_f90(material_params);

    let fresnel = fresnel_schlick(view_dot_halfway, f0, f90);

    (1.0 - fresnel) * distribution * geometric_shadowing * material_params.albedo_colour
}

pub struct IblVolumeRefractionParams {
    pub material_params: MaterialParams,
    pub framebuffer_size_x: u32,
    pub normal: Normal,
    pub view: View,
    pub proj_view_matrix: Mat4,
    pub position: Vec3,
    pub thickness: f32,
    pub model_scale: f32,
    pub attenuation_distance: f32,
    pub attenuation_colour: Vec3,
}

fn refract(incident: Vec3, normal: Vec3, index_of_refraction: IndexOfRefraction) -> Vec3 {
    let eta = 1.0 / index_of_refraction.0;

    let n_dot_i = normal.dot(incident);

    let k = 1.0 - eta * eta * (1.0 - n_dot_i * n_dot_i);

    eta * incident - (eta * n_dot_i + k.sqrt()) * normal
}

fn get_volume_transmission_ray(
    normal: Normal,
    view: View,
    thickness: f32,
    index_of_refraction: IndexOfRefraction,
    model_scale: f32,
) -> (Vec3, f32) {
    let refraction = refract(-view.0, normal.0, index_of_refraction);
    let length = thickness * model_scale;
    (refraction.normalize() * length, length)
}

// Component-wise natural log (log e) of a vector.
fn ln(vector: Vec3) -> Vec3 {
    Vec3::new(vector.x.ln(), vector.y.ln(), vector.z.ln())
}

fn apply_volume_attenuation(
    transmitted_light: Vec3,
    transmission_distance: f32,
    attenuation_distance: f32,
    attenuation_colour: Vec3,
) -> Vec3 {
    if attenuation_distance == f32::INFINITY {
        transmitted_light
    } else {
        // Compute light attenuation using Beer's law.
        let attenuation_coefficient = -ln(attenuation_colour) / attenuation_distance;
        // Beer's law
        let transmittance = (-attenuation_coefficient * transmission_distance).exp();
        transmittance * transmitted_light
    }
}

pub fn ibl_volume_refraction<
    FSamp: Fn(Vec2, f32) -> Vec3,
    GSamp: Fn(f32, PerceptualRoughness) -> Vec2,
>(
    params: IblVolumeRefractionParams,
    framebuffer_sampler: FSamp,
    ggx_lut_sampler: GSamp,
) -> Vec3 {
    let IblVolumeRefractionParams {
        framebuffer_size_x,
        proj_view_matrix,
        position,
        normal,
        view,
        thickness,
        model_scale,
        attenuation_colour,
        attenuation_distance,
        material_params:
            MaterialParams {
                albedo_colour,
                metallic: _,
                perceptual_roughness,
                index_of_refraction,
                specular_colour: _,
                specular_factor: _,
            },
    } = params;

    let material_params = params.material_params;

    //let thickness = 1.0;
    //let perceptual_roughness = PerceptualRoughness(0.25);

    let (ray, ray_length) =
        get_volume_transmission_ray(normal, view, thickness, index_of_refraction, model_scale);
    let refracted_ray_exit = position + ray;

    let device_coords = proj_view_matrix * refracted_ray_exit.extend(1.0);
    let screen_coords = device_coords.xy() / device_coords.w;
    let texture_coords = (screen_coords + 1.0) / 2.0;

    let framebuffer_lod =
        (framebuffer_size_x as f32).log2() * perceptual_roughness.apply_ior(index_of_refraction).0;

    let transmitted_light = framebuffer_sampler(texture_coords, framebuffer_lod);
    let attenuated_colour = apply_volume_attenuation(
        transmitted_light,
        ray_length,
        attenuation_distance,
        attenuation_colour,
    );

    let normal_dot_view = normal.0.dot(view.0);
    let brdf = ggx_lut_sampler(normal_dot_view, perceptual_roughness);

    let f0 = calculate_combined_f0(material_params);
    let f90 = calculate_combined_f90(material_params);

    let specular_colour = f0 * brdf.x + f90 * brdf.y;

    (1.0 - specular_colour) * attenuated_colour * albedo_colour
}

fn diffuse_brdf(base: Vec3, fresnel: Vec3) -> Vec3 {
    // not sure if max_element is needed here:
    // https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_specular#implementation
    (1.0 - fresnel.max_element()) * FRAC_1_PI * base
}

pub fn specular_brdf(
    normal_dot_view: Dot<Normal, View>,
    normal_dot_light: Dot<Normal, Light>,
    normal_dot_halfway: Dot<Normal, Halfway>,
    actual_roughness: ActualRoughness,
    fresnel: Vec3,
) -> Vec3 {
    let distribution_function = d_ggx(normal_dot_halfway, actual_roughness);

    let geometric_shadowing =
        v_smith_ggx_correlated(normal_dot_view, normal_dot_light, actual_roughness);

    (distribution_function * geometric_shadowing) * fresnel
}

pub fn basic_brdf(params: BasicBrdfParams) -> BrdfResult {
    let BasicBrdfParams {
        normal,
        light,
        light_intensity,
        view,
        material_params,
    } = params;

    let actual_roughness = material_params.perceptual_roughness.as_actual_roughness();

    let halfway = Halfway::new(&view, &light);
    let normal_dot_halfway = Dot::new(&normal, &halfway);
    let normal_dot_view = Dot::new(&normal, &view);
    let normal_dot_light = Dot::new(&normal, &light);
    let view_dot_halfway = Dot::new(&view, &halfway);

    let c_diff = material_params.diffuse_colour();

    let f0 = calculate_combined_f0(material_params);
    let f90 = calculate_combined_f90(material_params);

    let fresnel = fresnel_schlick(view_dot_halfway, f0, f90);

    let diffuse = light_intensity * normal_dot_light.value * diffuse_brdf(c_diff, fresnel);
    let specular = light_intensity
        * normal_dot_light.value
        * specular_brdf(
            normal_dot_view,
            normal_dot_light,
            normal_dot_halfway,
            actual_roughness,
            fresnel,
        );

    BrdfResult { diffuse, specular }
}

pub fn calculate_combined_f0(material: MaterialParams) -> Vec3 {
    let dielectric_specular_f0 = material.index_of_refraction.to_dielectric_f0()
        * material.specular_colour
        * material.specular_factor;
    dielectric_specular_f0.lerp(material.albedo_colour, material.metallic)
}

pub fn calculate_combined_f90(material: MaterialParams) -> Vec3 {
    let dielectric_specular_f90 = Vec3::splat(material.specular_factor);
    dielectric_specular_f90.lerp(Vec3::ONE, material.metallic)
}

#[derive(Default)]
pub struct BrdfResult {
    pub diffuse: Vec3,
    pub specular: Vec3,
}

impl Add<BrdfResult> for BrdfResult {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            diffuse: self.diffuse + other.diffuse,
            specular: self.specular + other.specular,
        }
    }
}

pub fn compute_f0(
    metallic: f32,
    index_of_refraction: IndexOfRefraction,
    diffuse_colour: Vec3,
) -> Vec3 {
    // from:
    // https://google.github.io/filament/Filament.md.html#materialsystem/parameterization/remapping
    let dielectric_f0 = index_of_refraction.to_dielectric_f0();
    let metallic_f0 = diffuse_colour;

    (1.0 - metallic) * dielectric_f0 + metallic * metallic_f0
}

#[derive(Copy, Clone)]
pub struct GgxLutValues {
    weight: f32,
    bias: f32,
}

pub fn ggx_lut_lookup<GSamp: Fn(f32, PerceptualRoughness) -> Vec2>(
    normal: Normal,
    view: View,
    material_params: MaterialParams,
    ggx_lut_sampler: GSamp,
) -> GgxLutValues {
    let normal_dot_view = Dot::new(&normal, &view);

    let lut_values = ggx_lut_sampler(normal_dot_view.value, material_params.perceptual_roughness);

    GgxLutValues {
        weight: lut_values.x,
        bias: lut_values.y,
    }
}

pub fn ibl_irradiance_lambertian<DSamp: Fn(Vec3) -> Vec3>(
    normal: Normal,
    view: View,
    material_params: MaterialParams,
    lut_values: GgxLutValues,
    diffuse_cubemap_sampler: DSamp,
) -> Vec3 {
    let normal_dot_view = Dot::new(&normal, &view);

    let irradiance = diffuse_cubemap_sampler(normal.0);

    // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
    // Roughness dependent fresnel, from Fdez-Aguera

    let fss_ess = calculate_fss_ess(material_params, normal_dot_view, lut_values);

    // Multiple scattering, from Fdez-Aguera
    let multiplier = caulcate_fms_ems_plus_kd(material_params, fss_ess, lut_values);

    multiplier * irradiance
}

#[test]
fn irradiance_is_zero_with_smooth() {
    let normal = Normal(Vec3::Y);
    let view = View(Vec3::Y);

    let material_params = MaterialParams {
        albedo_colour: Vec3::ONE,
        metallic: 1.0,
        perceptual_roughness: PerceptualRoughness(0.0),
        index_of_refraction: IndexOfRefraction::default(),
        specular_colour: Vec3::ONE,
        specular_factor: 1.0,
    };

    let lut_values = GgxLutValues {
        weight: 1.0,
        bias: 0.0,
    };

    let mip_count = 0;
    let diffuse_cubemap_sampler = |_| Vec3::ONE;

    let val = ibl_irradiance_lambertian(
        normal,
        view,
        material_params,
        lut_values,
        diffuse_cubemap_sampler,
    );

    assert_eq!(val, Vec3::ZERO);
}

pub fn get_ibl_radiance_ggx<SSamp: Fn(Vec3, f32) -> Vec3>(
    normal: Normal,
    view: View,
    material_params: MaterialParams,
    lut_values: GgxLutValues,
    mip_count: u32,
    specular_cubemap_sampler: SSamp,
) -> Vec3 {
    let lod = material_params.perceptual_roughness.0 * (mip_count - 1) as f32;

    let normal_dot_view = Dot::new(&normal, &view);

    let reflection = reflect(-view.0, normal.0).normalize();

    let radiance = specular_cubemap_sampler(reflection, lod);

    let fss_ess = calculate_fss_ess(material_params, normal_dot_view, lut_values);

    radiance * fss_ess
}

// See https://bruop.github.io/ibl/ and
// 'A Multiple-Scattering Microfacet Model for Real-Time Image-based Lighting':
// https://www.jcgt.org/published/0008/01/03/paper.pdf
//
// Section 'Roughness Dependent Fresnel'
fn calculate_fss_ess(
    material_params: MaterialParams,
    normal_dot_view: Dot<Normal, View>,
    lut_values: GgxLutValues,
) -> Vec3 {
    let f0 = calculate_combined_f0(material_params);

    // Modified fresnel term. Not sure why it can't use the halfway vector.
    let f_r = Vec3::splat(1.0 - material_params.perceptual_roughness.0).max(f0) - f0;

    // Wierd AF undocumented term from the paper
    let k_s = f0 + f_r * (1.0 - normal_dot_view.value).powf(5.0);

    // The gltf sample viewer inserts the specular factor here.
    material_params.specular_factor * k_s * lut_values.weight + lut_values.bias
}

fn caulcate_fms_ems_plus_kd(
    material_params: MaterialParams,
    fss_ess: Vec3,
    lut_values: GgxLutValues,
) -> Vec3 {
    let f0 = calculate_combined_f0(material_params);

    let e_ss = lut_values.weight + lut_values.bias;

    let e_ms = 1.0 - e_ss;

    let f_avg = f0 + (1.0 - f0) / 21.0;

    let f_ms = fss_ess * f_avg / (1.0 - e_ms * f_avg);

    let fms_ems = f_ms * e_ms;

    let e_dss = 1.0 - fss_ess + fms_ems;

    let k_d = material_params.diffuse_colour() * e_dss;

    k_d + fms_ems
}
