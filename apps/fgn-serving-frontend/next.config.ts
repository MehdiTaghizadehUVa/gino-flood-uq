import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      {
        source: "/runs/:runId",
        destination: "/demo/runs/:runId",
        permanent: true
      },
      {
        source: "/runs",
        destination: "/demo?workspace=runs",
        permanent: true
      }
    ];
  }
};

export default nextConfig;
