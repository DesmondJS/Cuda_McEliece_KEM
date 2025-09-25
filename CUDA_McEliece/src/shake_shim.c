#include <openssl/evp.h>
#include <openssl/err.h>
#include <stddef.h>
#include <stdio.h>

int SHAKE256(unsigned char *out, size_t outlen,
             const unsigned char *in, size_t inlen)
{
    int ok = 0;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx) goto done;

#if OPENSSL_VERSION_NUMBER >= 0x10101000L
    if (EVP_DigestInit_ex(ctx, EVP_shake256(), NULL) == 1 &&
        EVP_DigestUpdate(ctx, in, inlen) == 1 &&
        EVP_DigestFinalXOF(ctx, out, outlen) == 1)
    {
        ok = 1;
    }
#endif

done:
    if (!ok) {
        unsigned long err;
        char buf[256];
        while ((err = ERR_get_error()) != 0) {
            ERR_error_string_n(err, buf, sizeof(buf));
            fprintf(stderr, "OpenSSL SHAKE256 error: %s\n", buf);
        }
    }
    EVP_MD_CTX_free(ctx);
    return ok ? 0 : 1;
}
