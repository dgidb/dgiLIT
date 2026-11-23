declare namespace Cloudflare {
  interface Env {
    DGILIT_API_ORIGIN?: string;
    DB?: D1Database;
    BUCKET?: R2Bucket;
  }
}
