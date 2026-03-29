use once_cell::sync::Lazy;
use reqwest::Client;
use std::time::Duration;

static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to build HTTP client")
});

static HTTP_CLIENT_NO_COMPRESSION: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .build()
        .expect("Failed to build HTTP client without compression")
});

pub fn get_http_client() -> &'static Client {
    &HTTP_CLIENT
}

pub fn get_http_client_no_compression() -> &'static Client {
    &HTTP_CLIENT_NO_COMPRESSION
}
