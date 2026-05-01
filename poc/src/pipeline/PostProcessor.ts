/**
 * Stage 4 — Post-Processing
 *
 * Two operations:
 *
 *  A. Poisson Blending (expo-gl / WebGL)
 *     Smooths hard colour discontinuities at mask boundary.
 *     Solves:  ∇²f = div(v)  inside Ω,  f = f*  on ∂Ω
 *     Where:  v = gradient of source (inpainted region)
 *             f* = target pixel values (original image)
 *     Implemented as iterative Jacobi on GPU via WebGL fragment shader.
 *     K iterations of:  f_new(p) = (f*(N,E,S,W) + rhs(p)) / 4
 *     where rhs encodes the divergence of the guidance field.
 *
 *  B. Quality Assessment
 *     Computes SSIM and PSNR on the masked region to decide
 *     whether to offer the user a "request higher quality" button
 *     that triggers the server-side LaMa pipeline.
 *
 *     SSIM(x,y) = (2μₓμᵧ+C₁)(2σₓᵧ+C₂) / ((μₓ²+μᵧ²+C₁)(σₓ²+σᵧ²+C₂))
 *     PSNR      = 10·log₁₀(255²/MSE)
 */

import { GLView } from 'expo-gl';
import type { ExpoWebGLRenderingContext } from 'expo-gl';
import type { MaskResult } from './MaskGenerator';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface PostProcessConfig {
  enablePoisson: boolean;     // default true
  poissonIterations: number;  // default 50 (GPU Jacobi steps)
  qualityThreshold: number;   // SSIM below this → prompt server upgrade, default 0.82
}

export interface QualityMetrics {
  ssim:      number;
  psnr:      number;
  mse:       number;
  aboveThreshold: boolean;
}

export interface PostProcessResult {
  outputUri:    string;
  quality:      QualityMetrics;
  poissonApplied: boolean;
  processingMs: number;
}

const DEFAULT_POST_CONFIG: PostProcessConfig = {
  enablePoisson: true,
  poissonIterations: 50,
  qualityThreshold: 0.82,
};

// ─── WebGL Poisson Blending ───────────────────────────────────────────────────

/** GLSL vertex shader — full-screen quad */
const VERT_SRC = `
  attribute vec2 a_pos;
  varying   vec2 v_uv;
  void main() {
    v_uv        = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
  }
`;

/**
 * GLSL fragment shader — one Jacobi iteration of Poisson solve.
 *
 * f_new(p) = [ f(N) + f(E) + f(S) + f(W)
 *              + rhs(p)    ← divergence of guidance field v
 *            ] / 4
 *
 * Boundary condition: where mask = 0, f stays locked to target (original).
 */
const FRAG_POISSON = `
  precision mediump float;

  uniform sampler2D u_current;    // current iterate f
  uniform sampler2D u_target;     // original image f*
  uniform sampler2D u_source;     // inpainted image g
  uniform sampler2D u_mask;       // binary mask (r=1 inside Ω)
  uniform vec2      u_texel;      // 1/width, 1/height

  varying vec2 v_uv;

  void main() {
    float inside = texture2D(u_mask, v_uv).r;

    if (inside < 0.5) {
      // Outside mask: lock to original
      gl_FragColor = texture2D(u_target, v_uv);
      return;
    }

    // Sample 4-neighbour current iterate
    vec4 fN = texture2D(u_current, v_uv + vec2( 0.0,  u_texel.y));
    vec4 fE = texture2D(u_current, v_uv + vec2( u_texel.x,  0.0));
    vec4 fS = texture2D(u_current, v_uv + vec2( 0.0, -u_texel.y));
    vec4 fW = texture2D(u_current, v_uv + vec2(-u_texel.x,  0.0));

    // Guidance field divergence: div(∇g) ≈ g(N)+g(E)+g(S)+g(W) - 4g(p)
    vec4 gC = texture2D(u_source, v_uv);
    vec4 gN = texture2D(u_source, v_uv + vec2( 0.0,  u_texel.y));
    vec4 gE = texture2D(u_source, v_uv + vec2( u_texel.x,  0.0));
    vec4 gS = texture2D(u_source, v_uv + vec2( 0.0, -u_texel.y));
    vec4 gW = texture2D(u_source, v_uv + vec2(-u_texel.x,  0.0));
    vec4 rhs = gN + gE + gS + gW - 4.0 * gC;

    // Jacobi update
    gl_FragColor = (fN + fE + fS + fW + rhs) / 4.0;
  }
`;

export class PoissonBlender {
  private gl: ExpoWebGLRenderingContext | null = null;

  setGL(gl: ExpoWebGLRenderingContext) {
    this.gl = gl;
  }

  /**
   * Run Poisson blending entirely on GPU.
   * Returns updated pixel buffer.
   */
  blend(
    originalPixels: Uint8Array,
    inpaintedPixels: Uint8Array,
    mask: MaskResult,
    iterations: number,
  ): Uint8Array {
    const gl = this.gl;
    if (!gl) {
      console.warn('[Poisson] No GL context — skipping blend');
      return inpaintedPixels;
    }

    const { width: W, height: H } = mask;

    // Compile shaders
    const prog = this.compileProgram(gl, VERT_SRC, FRAG_POISSON);
    if (!prog) return inpaintedPixels;

    // Upload textures
    const texTarget   = this.uploadTexture(gl, originalPixels,  W, H);
    const texSource   = this.uploadTexture(gl, inpaintedPixels, W, H);
    const texMask     = this.uploadTexture(gl, this.expandMask(mask.data, W, H), W, H);
    let   texCurrent  = this.uploadTexture(gl, inpaintedPixels, W, H);

    // Two framebuffer ping-pong
    const [fbo0, tex0] = this.makeFramebuffer(gl, W, H);
    const [fbo1, tex1] = this.makeFramebuffer(gl, W, H);

    // Full-screen quad
    const quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

    gl.useProgram(prog);
    gl.viewport(0, 0, W, H);

    const bindTex = (unit: number, tex: WebGLTexture, name: string) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(gl.getUniformLocation(prog, name)!, unit);
    };

    gl.uniform2f(gl.getUniformLocation(prog, 'u_texel')!, 1 / W, 1 / H);

    const posLoc = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Jacobi iterations (ping-pong between fbo0 and fbo1)
    let readTex  = texCurrent;
    let writeFBO = fbo0;
    let writeTex = tex0;

    for (let i = 0; i < iterations; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, writeFBO);
      bindTex(0, readTex,   'u_current');
      bindTex(1, texTarget, 'u_target');
      bindTex(2, texSource, 'u_source');
      bindTex(3, texMask,   'u_mask');
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // Swap
      [readTex, writeTex] = [writeTex, readTex];
      writeFBO = writeFBO === fbo0 ? fbo1 : fbo0;
    }

    // Read back result
    const result = new Uint8Array(W * H * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeFBO === fbo0 ? fbo1 : fbo0);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, result);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    return result;
  }

  private compileProgram(gl: ExpoWebGLRenderingContext, vert: string, frag: string): WebGLProgram | null {
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vert); gl.compileShader(vs);
    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, frag); gl.compileShader(fs);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs); gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      console.error('[Poisson] Shader link error:', gl.getProgramInfoLog(prog));
      return null;
    }
    return prog;
  }

  private uploadTexture(gl: ExpoWebGLRenderingContext, data: Uint8Array, w: number, h: number): WebGLTexture {
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    return tex;
  }

  private makeFramebuffer(gl: ExpoWebGLRenderingContext, w: number, h: number): [WebGLFramebuffer, WebGLTexture] {
    const tex = this.uploadTexture(gl, new Uint8Array(w * h * 4), w, h);
    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return [fbo, tex];
  }

  private expandMask(mask: Uint8Array, w: number, h: number): Uint8Array {
    const rgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      const v = mask[i] > 0 ? 255 : 0;
      rgba[i*4] = v; rgba[i*4+1] = v; rgba[i*4+2] = v; rgba[i*4+3] = 255;
    }
    return rgba;
  }
}

// ─── Quality Metrics ──────────────────────────────────────────────────────────

export class QualityAssessor {
  /**
   * Compute SSIM and PSNR between original and result
   * only over the masked region to avoid rewarding unchanged pixels.
   *
   * SSIM uses 11×11 Gaussian window; simplified to global stats for mobile perf.
   */
  assess(
    original: Uint8Array,
    result:   Uint8Array,
    mask:     MaskResult,
    threshold: number,
  ): QualityMetrics {
    const W = mask.width;
    const N = mask.width * mask.height;

    let sumA = 0, sumB = 0, sumA2 = 0, sumB2 = 0, sumAB = 0, mse = 0;
    let count = 0;

    for (let i = 0; i < N; i++) {
      if (mask.data[i] === 0) continue; // only assess masked region
      count++;

      // Use luminance: Y = 0.299R + 0.587G + 0.114B
      const lumA = (original[i*4] * 0.299 + original[i*4+1] * 0.587 + original[i*4+2] * 0.114) / 255;
      const lumB = (result[i*4]   * 0.299 + result[i*4+1]   * 0.587 + result[i*4+2]   * 0.114) / 255;

      sumA  += lumA;  sumB  += lumB;
      sumA2 += lumA * lumA;  sumB2 += lumB * lumB;
      sumAB += lumA * lumB;
      mse   += (lumA - lumB) ** 2;
    }

    if (count === 0) return { ssim: 1, psnr: Infinity, mse: 0, aboveThreshold: true };

    const muA   = sumA  / count;
    const muB   = sumB  / count;
    const sigA2 = sumA2 / count - muA * muA;
    const sigB2 = sumB2 / count - muB * muB;
    const sigAB = sumAB / count - muA * muB;
    const mseV  = mse / count;

    // SSIM constants: C1=(0.01)², C2=(0.03)²
    const C1 = 0.0001, C2 = 0.0009;
    const ssim = ((2 * muA * muB + C1) * (2 * sigAB + C2))
               / ((muA**2 + muB**2 + C1) * (sigA2 + sigB2 + C2));

    const psnr = mseV === 0 ? Infinity : 10 * Math.log10(1 / mseV);

    return {
      ssim:  Math.max(0, Math.min(1, ssim)),
      psnr:  Math.round(psnr * 100) / 100,
      mse:   Math.round(mseV * 1e6) / 1e6,
      aboveThreshold: ssim >= threshold,
    };
  }
}

export const poissonBlender = new PoissonBlender();
export const qualityAssessor = new QualityAssessor();
